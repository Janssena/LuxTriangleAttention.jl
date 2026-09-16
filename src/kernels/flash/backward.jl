# The reverse of `flash_fwd!` (see `forward.jl`), the same three constraints (1-D group dispatch, raw
# pointers not device arrays, Float32 accumulation) all apply here unchanged.
# 
# None of the key-side kernels touch `dBias` — `flash_bwd_dq!` already covers every `(kk, js, bh)`
# once per `i`, and accumulating it again would double-count it.

"""
    flash_bwd_dq!(dq, dbias, q, k, v, bias, mask, dO, lse, delta,
        scale, n, nouter, nh, mheads, Val(T), Val(DIM), Val(TG), Val(KT))

Backward pass, `dQ`/`dBias` half: one thread per query, the same grid geometry as [`flash_fwd!`](@ref)
(and the same tile-staging of `k`/`v`).

`dQ[:, js, i, bh]` is written once, at the end, by the thread that owns `js`, no atomics. `dBias`
is accumulated across the key-tile loop via [`_atomic_fadd!`](@ref): the address `bb + kk + 1` is
shared by every `i` (bias has no `i` axis), so `nouter` threads across different groups can land on it,
but never more than `nouter`, queries and keys otherwise sole-own their contribution.

`dbias` MUST be zeroed by the caller before launch; `dq` need not be (every active thread's location
is fully overwritten, exactly as the forward writes `o`).
"""
@kernel unsafe_indices = true function flash_bwd_dq!(dq, dbias, q, k, v, bias, mask, dO, lse, delta,
    scale, n, nouter, nh, mheads, ::Val{T}, ::Val{DIM}, ::Val{TG}, ::Val{KT}) where {T,DIM,TG,KT}
    ks = @localmem T (DIM, KT)
    vs = @localmem T (DIM, KT)

    gx, gy, gz = @index(Group, NTuple)
    tid = Int32(@index(Local, Linear))

    jt = Int32(gx) - Int32(1)
    i = Int32(gy) - Int32(1)
    bh = Int32(gz) - Int32(1)
    j = jt * Int32(TG) + tid - Int32(1)

    active = j < n
    js = active ? j : Int32(0)

    hi = bh % nh
    bi = bh ÷ nh
    nn = n * n
    row = (bi * nouter + i) * n
    hstep = nh * Int32(DIM)
    qb = (row + js) * hstep + hi * Int32(DIM)
    kb = row * hstep + hi * Int32(DIM)
    bb = bh * nn + js * n
    mb = ((bh ÷ mheads) * nouter + i) * n     # see `flash_fwd!` for why this is not `bi * nn`
    lb = ((bi * nouter + i) * nh + hi) * n + j + Int32(1)

    qv = _kload(q, qb, Val(DIM))
    dov = _kload(dO, qb, Val(DIM))
    lse_j = active ? unsafe_load(lse, lb) : zero(Float32)
    delta_j = active ? unsafe_load(delta, lb) : zero(Float32)

    dqv = ntuple(Returns(zero(Float32)), Val(DIM))

    t = Int32(0)
    while t < n
        @synchronize
        e = tid
        while e <= Int32(KT) * Int32(DIM)
            d0 = (e - Int32(1)) % Int32(DIM)
            c0 = (e - Int32(1)) ÷ Int32(DIM)
            kk = t + c0
            vk = kk < n ? unsafe_load(k, kb + kk * hstep + d0 + Int32(1)) : zero(T)
            vv = kk < n ? unsafe_load(v, kb + kk * hstep + d0 + Int32(1)) : zero(T)
            @inbounds ks[d0+Int32(1), c0+Int32(1)] = vk
            @inbounds vs[d0+Int32(1), c0+Int32(1)] = vv
            e += Int32(TG)
        end
        @synchronize

        if active
            for c in Int32(1):Int32(KT)
                kk = t + c - Int32(1)
                if kk < n
                    @inbounds masked = unsafe_load(mask, mb + kk + Int32(1))
                    if !masked
                        s = _kdot(qv, ks, c) * scale +
                            Float32(unsafe_load(bias, bb + kk + Int32(1)))
                        p = exp(s - lse_j)
                        dp = _kdot(dov, vs, c)
                        ds = p * (dp - delta_j)
                        dqv = _kaccum(dqv, ds * scale, ks, c)
                        _atomic_fadd!(dbias, bb + kk + Int32(1), ds)
                    end
                end
            end
        end
        t += Int32(KT)
    end

    # `_kstore!` (from `forward.jl`) rather than an inline `ntuple(...) do d ... end` closure here: a
    # closure capturing a kernel argument (`dq`, `qb`) loses its type under `@kernel` and fails to
    # compile with "unsupported dynamic function invocation", see `_kdot`'s docstring in `forward.jl`.
    # `inv_l = one(Float32)` makes it a plain store; the backward has nothing to divide by.
    active && _kstore!(dq, dqv, one(Float32), qb, T)
end

"""
    flash_bwd_dk!(dk, q, k, v, bias, mask, dO, lse, delta,
        scale, n, nouter, nh, mheads, Val(T), Val(DIM), Val(TG), Val(KT))

Backward pass, `dK`, one thread per KEY, streaming query tiles. The transposed geometry of
[`flash_bwd_dq!`](@ref): `q`/`dO` (plus `lse`/`delta`) are staged into local memory instead of
`k`/`v`, and the thread-resident registers hold this key's own `k`/`v` instead of `q`.

`dK[:, kk, i, bh]` is written once, by the thread that owns `kk`, no atomics, same reasoning as
`dQ` above. Does NOT touch `dBias`; see the file header.

Holds `kv`, `vv` and the `dkv` accumulator, `3*DIM` registers, the same footprint as
[`flash_bwd_dq!`](@ref). `vv` is resident here, and absent from [`flash_bwd_dv!`](@ref), because
`dK` alone needs `dp = v·dO` in order to form `ds`. That asymmetry is the entire reason the two are
separate kernels rather than one: see the register-pressure note in the file header.

A masked key contributes nothing at any query (in the forward, `p = 0` there for every column), so
`dK` is zero for it, applied as a zero factor at the store (see `keep` in the body for why not a
branch), and never by an early return, since every thread in a group must still take part in the
cooperative staging and its
`@synchronize` calls regardless of its own `mask` value (masking is per-thread here, unlike the
forward and `flash_bwd_dq!`, where it varies per loop ITERATION of a single thread: see the file
header for why an uneven `@synchronize` count is not just wrong but unsafe).
"""
@kernel unsafe_indices = true function flash_bwd_dk!(dk, q, k, v, bias, mask, dO, lse, delta,
    scale, n, nouter, nh, mheads, ::Val{T}, ::Val{DIM}, ::Val{TG}, ::Val{KT}) where {T,DIM,TG,KT}
    qs = @localmem T (DIM, KT)
    dos = @localmem T (DIM, KT)
    lses = @localmem Float32 (KT,)
    deltas = @localmem Float32 (KT,)

    gx, gy, gz = @index(Group, NTuple)
    tid = Int32(@index(Local, Linear))

    ktile = Int32(gx) - Int32(1)
    i = Int32(gy) - Int32(1)
    bh = Int32(gz) - Int32(1)
    kk = ktile * Int32(TG) + tid - Int32(1)

    active = kk < n
    kks = active ? kk : Int32(0)

    hi = bh % nh
    bi = bh ÷ nh
    nn = n * n
    row = (bi * nouter + i) * n
    hstep = nh * Int32(DIM)
    kb = (row + kks) * hstep + hi * Int32(DIM)
    qb0 = row * hstep + hi * Int32(DIM)
    mb = ((bh ÷ mheads) * nouter + i) * n     # see `flash_fwd!` for why this is not `bi * nn`
    lrow = (bi * nouter + i) * nh + hi

    masked = active ? unsafe_load(mask, mb + kks + Int32(1)) : true
    # A masked key's gradient is ZERO, applied as a factor at the store rather than by skipping the
    # loop body under `if active & !masked`. Measured: the data-dependent branch made this kernel
    # 15-25% slower once the mask base involved the runtime `nouter` (the compiler's scheduling
    # around it degrades), while the factor form is ~1% faster than the pre-`nouter` kernel on dense
    # input and bit-identical. The cost moves to masked keys, which now do the arithmetic and
    # discard it — only visible when a WHOLE SIMD group of keys is masked (long padding tails).
    keep = (active & !masked) ? one(Float32) : zero(Float32)

    kv = _kload(k, kb, Val(DIM))
    vv = _kload(v, kb, Val(DIM))
    dkv = ntuple(Returns(zero(Float32)), Val(DIM))

    t = Int32(0)
    while t < n
        @synchronize
        e = tid
        while e <= Int32(KT) * Int32(DIM)
            d0 = (e - Int32(1)) % Int32(DIM)
            c0 = (e - Int32(1)) ÷ Int32(DIM)
            jj = t + c0
            vq = jj < n ? unsafe_load(q, qb0 + jj * hstep + d0 + Int32(1)) : zero(T)
            vdo = jj < n ? unsafe_load(dO, qb0 + jj * hstep + d0 + Int32(1)) : zero(T)
            @inbounds qs[d0+Int32(1), c0+Int32(1)] = vq
            @inbounds dos[d0+Int32(1), c0+Int32(1)] = vdo
            # `lse`/`delta` are per-(query,i,bh) scalars: load them once per column, on the thread
            # handling that column's first channel, rather than once per (channel, column).
            if d0 == Int32(0)
                jjl = jj < n ? jj + Int32(1) : Int32(1)
                @inbounds lses[c0+Int32(1)] = jj < n ? unsafe_load(lse, lrow * n + jjl) : zero(Float32)
                @inbounds deltas[c0+Int32(1)] = jj < n ? unsafe_load(delta, lrow * n + jjl) : zero(Float32)
            end
            e += Int32(TG)
        end
        @synchronize

        if active
            for c in Int32(1):Int32(KT)
                jj = t + c - Int32(1)
                if jj < n
                    s = _kdot(kv, qs, c) * scale +
                        Float32(unsafe_load(bias, bh * nn + jj * n + kks + Int32(1)))
                    p = exp(s - lses[c])
                    dp = _kdot(vv, dos, c)
                    ds = p * (dp - deltas[c])
                    dkv = _kaccum(dkv, ds * scale, qs, c)
                end
            end
        end
        t += Int32(KT)
    end

    active && _kstore!(dk, dkv, keep, kb, T)
end

"""
    flash_bwd_dv!(dv, q, k, bias, mask, dO, lse,
        scale, n, nouter, nh, mheads, Val(T), Val(DIM), Val(TG), Val(KT))

Backward pass, `dV`, one thread per KEY, streaming query tiles, the same geometry and masking
rules as [`flash_bwd_dk!`](@ref).

`dV[:, kk] = Σ_j p_{j,kk} dO_j` needs only the softmax weight `p`, never `dp` and never `delta`.
So this kernel takes NEITHER `v` NOR `delta` as an argument, they are absent from the signature
rather than merely unused, so nothing binds a buffer the kernel never reads, and it holds only
`kv` plus the `dvv` accumulator: `2*DIM` registers, the same footprint as `flash_fwd!`, the one
kernel in this file that does not spill.

That is what the split buys, and it is why `dV` rather than `dK` is the half worth separating; see
the register-pressure note in the file header. The `s`/`p` recomputed here duplicates
[`flash_bwd_dk!`](@ref)'s, which is pure ALU and repaid many times over by not spilling.
"""
@kernel unsafe_indices = true function flash_bwd_dv!(dv, q, k, bias, mask, dO, lse,
    scale, n, nouter, nh, mheads, ::Val{T}, ::Val{DIM}, ::Val{TG}, ::Val{KT}) where {T,DIM,TG,KT}
    qs = @localmem T (DIM, KT)
    dos = @localmem T (DIM, KT)
    lses = @localmem Float32 (KT,)

    gx, gy, gz = @index(Group, NTuple)
    tid = Int32(@index(Local, Linear))

    ktile = Int32(gx) - Int32(1)
    i = Int32(gy) - Int32(1)
    bh = Int32(gz) - Int32(1)
    kk = ktile * Int32(TG) + tid - Int32(1)

    active = kk < n
    kks = active ? kk : Int32(0)

    hi = bh % nh
    bi = bh ÷ nh
    nn = n * n
    row = (bi * nouter + i) * n
    hstep = nh * Int32(DIM)
    kb = (row + kks) * hstep + hi * Int32(DIM)
    qb0 = row * hstep + hi * Int32(DIM)
    mb = ((bh ÷ mheads) * nouter + i) * n     # see `flash_fwd!` for why this is not `bi * nn`
    lrow = (bi * nouter + i) * nh + hi

    masked = active ? unsafe_load(mask, mb + kks + Int32(1)) : true
    # A masked key's gradient is ZERO, applied as a factor at the store rather than by skipping the
    # loop body under `if active & !masked`. Measured: the data-dependent branch made this kernel
    # 15-25% slower once the mask base involved the runtime `nouter` (the compiler's scheduling
    # around it degrades), while the factor form is ~1% faster than the pre-`nouter` kernel on dense
    # input and bit-identical. The cost moves to masked keys, which now do the arithmetic and
    # discard it — only visible when a WHOLE SIMD group of keys is masked (long padding tails).
    keep = (active & !masked) ? one(Float32) : zero(Float32)

    kv = _kload(k, kb, Val(DIM))
    dvv = ntuple(Returns(zero(Float32)), Val(DIM))

    t = Int32(0)
    while t < n
        @synchronize
        e = tid
        while e <= Int32(KT) * Int32(DIM)
            d0 = (e - Int32(1)) % Int32(DIM)
            c0 = (e - Int32(1)) ÷ Int32(DIM)
            jj = t + c0
            vq = jj < n ? unsafe_load(q, qb0 + jj * hstep + d0 + Int32(1)) : zero(T)
            vdo = jj < n ? unsafe_load(dO, qb0 + jj * hstep + d0 + Int32(1)) : zero(T)
            @inbounds qs[d0+Int32(1), c0+Int32(1)] = vq
            @inbounds dos[d0+Int32(1), c0+Int32(1)] = vdo
            if d0 == Int32(0)
                jjl = jj < n ? jj + Int32(1) : Int32(1)
                @inbounds lses[c0+Int32(1)] = jj < n ? unsafe_load(lse, lrow * n + jjl) : zero(Float32)
            end
            e += Int32(TG)
        end
        @synchronize

        if active
            for c in Int32(1):Int32(KT)
                jj = t + c - Int32(1)
                if jj < n
                    s = _kdot(kv, qs, c) * scale +
                        Float32(unsafe_load(bias, bh * nn + jj * n + kks + Int32(1)))
                    p = exp(s - lses[c])
                    dvv = _kaccum(dvv, p, dos, c)
                end
            end
        end
        t += Int32(KT)
    end

    active && _kstore!(dv, dvv, keep, kb, T)
end

"""
    flash_bwd_delta(o, dO) -> Array{Float32}

`delta[j,i,bh] = Σ_d o[d,j,i,bh] * dO[d,j,i,bh]`, shape matching `lse`, the row constant both
backward kernels need (`Σ_j p_{j,kk}(dP_{j,kk} - delta_j) = 0` is the softmax-Jacobian identity that
makes `dS = p * (dP - delta)` correct; see e.g. the FlashAttention-2 paper).

Ordinary broadcasting, not a `@kernel`: the RESULT costs nothing extra to materialise (it is exactly
the shape of `o`/`dO` reduced over one axis already the same size as `lse`), so a raw-pointer kernel
buys nothing here. Works on any array backend — CPU, `MtlArray`, whatever `q`/`k`/`v` already are,
for the same reason `_sdpa_kernel`'s CPU fallback needs no device-specific kernel of its own either,
see `src/kernels/backend.jl`.

Written as a FUSED reduction rather than the obvious `sum(Float32.(o) .* Float32.(dO); dims=1)`,
because that form materialises the PRODUCT in full before reducing it away: one extra
`[dim, n, n, bh]` buffer, 134 MB at N=512. Fusing is also ~3.4x faster, since the temporary is 
never written and read back.

Two details that are easy to get wrong:

  * The broadcast must be `instantiate`d. `mapreducedim!` calls `check_reducedims`, which asks the
    source for `axes(A, i)`, and an un-instantiated `Broadcasted` has no axes yet.
  * The multiply is done in `Float32` explicitly, via the lambda, rather than by broadcasting
    `Float32` over each operand first. For `T = Float32` these agree, but for a narrower storage
    type `broadcasted(*, o, dO)` would multiply in `T` and only widen on the way into the
    accumulator, silently losing the precision the file header requires (see note 3 in
    `forward.jl`). Both operands are widened BEFORE the multiply, exactly as the unfused version
    did. Still lazy, so this costs no memory.

`Base.mapreducedim!` accepts a `Broadcasted` source on CPU and GPU alike (Base's own
`AbstractArrayOrBroadcasted`), so the fusion does not cost portability.
"""
function flash_bwd_delta(o::AbstractArray{T,4}, dO) where {T}
    r = similar(o, Float32, (1, size(o, 2), size(o, 3), size(o, 4)))
    fill!(r, zero(Float32))
    bc = Base.Broadcast.instantiate(
        Base.broadcasted((a, b) -> Float32(a) * Float32(b), o, dO))
    Base.mapreducedim!(identity, Base.add_sum, r, bc)
    return dropdims(r; dims=1)
end


"""
    _atomic_fadd!(ptr, idx, val)

Atomically add `val` into element `idx` (1-based, the same convention `unsafe_load`/`unsafe_store!`
use throughout this file) of the `Float32` buffer `ptr` points to.

Declared here with no implementation. An atomic add has no portable expression over
`KernelAbstractions`'s raw-pointer kernels (see note 2 in `forward.jl` for why raw pointers are used
at all), so a backend extension supplies the actual instruction for its own pointer type, e.g. the
Metal extension adds a method for `LLVMPtr{Float32,AS.Device}` built on
`Metal.atomic_fetch_add_explicit`. This is ordinary dispatch on a function this package owns, not
the `Ref`-based seam `device.jl` uses for `_DEVICE_BINARY`/`_KA_BACKEND_OBJECT`: those exist because
the Reactant extension needs something the Metal extension defines and cannot import it directly,
but here only ONE extension (whichever backend is loaded) ever needs to supply a method, so a plain
generic function is enough.
"""
function _atomic_fadd! end

"""Plain accumulate `accv += w * ks[:,c]`, no rescale: the backward never carries a running max, so
there is nothing to correct for."""
@inline _kaccum(accv::NTuple{D,Float32}, w, ks, c) where {D} =
    ntuple(d -> (@inbounds accv[d] + w * Float32(ks[d, c])), Val(D))

"""
    kernel_signature_dq(ctx_type, PT, PL, PB, T, dim, tg, kt) -> Type

Argument tuple type for ahead-of-time compilation of [`flash_bwd_dq!`](@ref), the `dQ`/`dBias`
pass. Mirrors [`kernel_signature`](@ref)'s role for the forward; `PL` (a pointer to `Float32`) is
reused for both `dbias` and `lse`/`delta`, since all three are the never-narrowed accumulation type.
"""
kernel_signature_dq(::Type{C}, ::Type{PT}, ::Type{PL}, ::Type{PB},
    ::Type{T}, dim, tg, kt) where {C,PT,PL,PB,T} =
    Tuple{C,PT,PL,PT,PT,PT,PT,PB,PT,PL,PL,Float32,Int32,Int32,Int32,Int32,
        Val{T},Val{dim},Val{tg},Val{kt}}

"""
    kernel_signature_dk(ctx_type, PT, PL, PB, T, dim, tg, kt) -> Type

Argument tuple type for ahead-of-time compilation of [`flash_bwd_dk!`](@ref), the `dK` pass.
Takes the same operands as [`kernel_signature_dq`](@ref) but returns ONE output rather than two.
"""
kernel_signature_dk(::Type{C}, ::Type{PT}, ::Type{PL}, ::Type{PB},
    ::Type{T}, dim, tg, kt) where {C,PT,PL,PB,T} =
    Tuple{C,PT,PT,PT,PT,PT,PB,PT,PL,PL,Float32,Int32,Int32,Int32,Int32,
        Val{T},Val{dim},Val{tg},Val{kt}}

"""
    kernel_signature_dv(ctx_type, PT, PL, PB, T, dim, tg, kt) -> Type

Argument tuple type for ahead-of-time compilation of [`flash_bwd_dv!`](@ref), the `dV` pass.

Shorter than the others by two pointers: `dV` needs neither `v` nor `delta` (see that kernel's
docstring), and they are omitted from the signature rather than passed and ignored, an argument the
kernel never reads would still occupy a buffer slot, and every caller's binding table is built by
POSITION, so a silently unused slot is a silently wrong table waiting to happen.
"""
kernel_signature_dv(::Type{C}, ::Type{PT}, ::Type{PL}, ::Type{PB},
    ::Type{T}, dim, tg, kt) where {C,PT,PL,PB,T} =
    Tuple{C,PT,PT,PT,PT,PB,PT,PL,Float32,Int32,Int32,Int32,Int32,
        Val{T},Val{dim},Val{tg},Val{kt}}
