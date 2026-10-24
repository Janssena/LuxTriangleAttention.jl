# The reverse of `flash_fwd!` (see `forward.jl`), the same four constraints (1-D group dispatch, raw
# pointers not device arrays, Float32 accumulation, no runtime-indexed register tuples) all apply
# here unchanged, and so do its base-2 softmax and its additive staged key mask.
#
# None of the key-side kernels touch `dBias` — `flash_bwd_dq!` already covers every `(kk, js, bh)`
# once per `i`, and accumulating it again would double-count it.
#
# ## Register pressure, and why there are three kernels
#
# Every thread keeps one row's vectors in registers: `flash_bwd_dq!` holds `q`, `dO` and the `dQ`
# accumulator, `flash_bwd_dk!` holds `k`, `v` and `dK` — `3*DIM` floats each — and
# `flash_bwd_dv!` holds `k` and `dV`, `2*DIM`, like the forward. Merging dK and dV into one kernel
# would need `4*DIM`. On an M1 Pro the `3*DIM` kernels are the ones that hit the register ceiling at
# `DIM = 32` (measured: dQ ~5x and dK ~3x the time of dV there, against ~1.5x at `DIM = 24`), so the
# split stays. Moving the accumulator out of registers instead — `ds` staged to local memory and
# folded into the OUTPUT buffer once per tile — was measured too, and is slower still. A device with
# a larger register file does not have this ceiling; the tuner (`tune.jl`) picks each stage's
# `(TG, KT)` independently so each kernel gets the geometry that suits its own footprint.
#
# ## Absent bias or mask
#
# Every kernel here takes the same compile-time `HB` (has bias) / `HM` (has mask) flags as
# `flash_fwd!` (see its docstring). `HB = false` drops the bias term from every recomputed score
# and, in `flash_bwd_dq!`, the `dbias` atomic as well; `HM = false` treats every in-range key as
# live without reading `mask`. The pointers stay in the signature, so the ABI does not change.
#
# ## Bias layout per kernel
#
# `flash_bwd_dq!` owns one QUERY per thread, like the forward, so it reads the TRANSPOSED bias
# (`:bias_t`, queries fastest) and accumulates `dBias` into a transposed buffer (`:dbias_t`) — both
# coalesced across a SIMD group. `flash_bwd_dk!` and `flash_bwd_dv!` own one KEY per thread, for which
# the ordinary keys-fastest `:bias` is already the coalesced layout, so they keep it.

"""
    flash_bwd_dq!(dq, dbias_t, q, k, v, bias_t, mask, dO, lse, delta,
        scale, n, nouter, nh, mheads, Val(T), Val(DIM), Val(TG), Val(KT), Val(HB), Val(HM))

Backward pass, `dQ`/`dBias` half: one thread per query, the same grid geometry as [`flash_fwd!`](@ref)
(and the same tile-staging of `k`/`v` and of the key mask).

`dQ[:, js, i, bh]` is written once, at the end, by the thread that owns `js`, no atomics. `dBias`
is accumulated across the key-tile loop via [`_atomic_fadd!`](@ref): the address is shared by every
`i` (bias has no `i` axis), so `nouter` threads across different groups can land on it, but never
more than `nouter`, queries and keys otherwise sole-own their contribution. `dbias_t` is the
gradient of `bias_t`, i.e. in the transposed `[n (queries), n (keys), bh]` layout, which is what
makes a SIMD group's 32 atomics land on 32 consecutive addresses.

A dead key (masked, or past `n`) is skipped outright. Its staged mask term is shared by the whole
group, so the branch is uniform — no divergence — and it saves the atomic as well as the arithmetic.

`dbias_t` MUST be zeroed by the caller before launch; `dq` need not be (every active thread's
location is fully overwritten, exactly as the forward writes `o`). With `HB = false` neither
`bias_t` nor `dbias_t` is touched.
"""
@kernel unsafe_indices = true function flash_bwd_dq!(dq, dbias_t, q, k, v, bias_t, mask, dO, lse, delta,
    scale, n, nouter, nh, mheads, ::Val{T}, ::Val{DIM}, ::Val{TG}, ::Val{KT}, ::Val{HB},
    ::Val{HM}) where {T,DIM,TG,KT,HB,HM}
    ks = @localmem T (DIM, KT)
    vs = @localmem T (DIM, KT)
    ms = @localmem Float32 (KT,)

    tid = _i32(@index(Local, Linear))

    # (query or key tile, outer index, batch * head), 0-based, decoded by hand from the linear group index
    jt, i, bh = _group_ids(_i32(@index(Group, Linear)) - Int32(1), n, nouter, Val(TG))
    j = jt * Int32(TG) + tid - Int32(1)

    active = j < n
    js = active ? j : Int32(0)

    hi = _srem(bh, nh)
    bi = _sdiv(bh, nh)
    nn = n * n
    row = (bi * nouter + i) * n
    hstep = nh * Int32(DIM)
    qb = (row + js) * hstep + hi * Int32(DIM)
    kb = row * hstep + hi * Int32(DIM)
    btb = bh * nn + js                        # bias_t / dbias_t: (js, kk) at btb + kk * n
    mb = (_sdiv(bh, mheads) * nouter + i) * n     # see `flash_fwd!` for why this is not `bi * nn`
    lb = ((bi * nouter + i) * nh + hi) * n + js + Int32(1)

    qv = _kscale(_kload(q, qb, Val(DIM)), scale * _LOG2E)
    dov = _kload(dO, qb, Val(DIM))
    lse2 = unsafe_load(lse, lb) * _LOG2E      # base 2, like the scores
    delta_j = unsafe_load(delta, lb)

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
        # `1` marks a dead key here, `0` a live one: only the live test below reads it.
        _kstage_mask!(ms, mask, mb, t, n, tid, one(Float32), Val(TG), Val(KT), Val(HM))
        @synchronize

        if active
            for c in Int32(1):Int32(KT)
                if @inbounds ms[c] == zero(Float32)
                    kk = t + c - Int32(1)
                    s2 = _kdot(qv, ks, c) + _kbias(bias_t, btb + kk * n + Int32(1), Val(HB))
                    p = exp2(s2 - lse2)
                    ds = p * (_kdot(dov, vs, c) - delta_j)
                    dqv = _kaccum(dqv, ds * scale, ks, c)
                    HB && _atomic_fadd!(dbias_t, btb + kk * n + Int32(1), ds)
                end
            end
        end
        t += Int32(KT)
    end

    active && _kstore!(dq, dqv, one(Float32), qb, T)
end

"""
    flash_bwd_dk!(dk, q, k, v, bias, mask, dO, lse, delta,
        scale, n, nouter, nh, mheads, Val(T), Val(DIM), Val(TG), Val(KT), Val(HB), Val(HM))

Backward pass, `dK`, one thread per KEY, streaming query tiles. The transposed geometry of
[`flash_bwd_dq!`](@ref): `q`/`dO` (plus `lse`/`delta`) are staged into local memory instead of
`k`/`v`, and the thread-resident registers hold this key's own `k`/`v` instead of `q`. `bias` is the
ordinary keys-fastest layout, which is the coalesced one for a thread per key.

`dK[:, kk, i, bh]` is written once, by the thread that owns `kk`, no atomics, same reasoning as
`dQ` above. Does NOT touch `dBias`; see the file header.

Holds `kv`, `vv` and the `dkv` accumulator, `3*DIM` registers, the same footprint as
[`flash_bwd_dq!`](@ref). `vv` is resident here, and absent from [`flash_bwd_dv!`](@ref), because
`dK` alone needs `dp = v·dO` in order to form `ds`. That asymmetry is the entire reason the two are
separate kernels rather than one: see the register-pressure note in the file header.

A masked key contributes nothing at any query of a slice that has a live key (in the forward, `p = 0` there for
every column; a slice with none attended uniformly, and `dV` gives every key its share, see `flash_bwd_dv!`), so
`dK` is written as ZERO for it — selected at the store, not multiplied in, so a non-finite
accumulator (a fully-masked row's `lse = -Inf`) cannot leak through as `0 * Inf = NaN` — and never
by an early return, since every thread in a group must still take part in the cooperative staging
and its `@synchronize` calls regardless of its own `mask` value. A query past `n` in a ragged tile
has its staged `lse` set to `+floatmax`, which makes its `p` exactly zero: no branch in the loop.
"""
@kernel unsafe_indices = true function flash_bwd_dk!(dk, q, k, v, bias, mask, dO, lse, delta,
    scale, n, nouter, nh, mheads, ::Val{T}, ::Val{DIM}, ::Val{TG}, ::Val{KT}, ::Val{HB},
    ::Val{HM}) where {T,DIM,TG,KT,HB,HM}
    qs = @localmem T (DIM, KT)
    dos = @localmem T (DIM, KT)
    lses = @localmem Float32 (KT,)
    deltas = @localmem Float32 (KT,)

    tid = _i32(@index(Local, Linear))

    # (query or key tile, outer index, batch * head), 0-based, decoded by hand from the linear group index
    ktile, i, bh = _group_ids(_i32(@index(Group, Linear)) - Int32(1), n, nouter, Val(TG))
    kk = ktile * Int32(TG) + tid - Int32(1)

    active = kk < n
    kks = active ? kk : Int32(0)

    hi = _srem(bh, nh)
    bi = _sdiv(bh, nh)
    nn = n * n
    row = (bi * nouter + i) * n
    hstep = nh * Int32(DIM)
    kb = (row + kks) * hstep + hi * Int32(DIM)
    qb0 = row * hstep + hi * Int32(DIM)
    mb = (_sdiv(bh, mheads) * nouter + i) * n     # see `flash_fwd!` for why this is not `bi * nn`
    lrow = (bi * nouter + i) * nh + hi

    masked = active ? (HM ? unsafe_load(mask, mb + kks + Int32(1)) : false) : true
    live = active & !masked

    kv = _kscale(_kload(k, kb, Val(DIM)), scale * _LOG2E)
    vv = _kload(v, kb, Val(DIM))
    dkv = ntuple(Returns(zero(Float32)), Val(DIM))

    t = Int32(0)
    while t < n
        @synchronize
        _kstage_rows!(qs, dos, q, dO, qb0, hstep, t, n, tid, Val(T), Val(DIM), Val(TG), Val(KT))
        _kstage_stats!(lses, deltas, lse, delta, lrow * n, t, n, tid, Val(TG), Val(KT))
        @synchronize

        for c in Int32(1):Int32(KT)
            jj = min(t + c - Int32(1), n - Int32(1))
            s2 = _kdot(kv, qs, c) + _kbias(bias, bh * nn + jj * n + kks + Int32(1), Val(HB))
            p = @inbounds exp2(s2 - lses[c])
            ds = @inbounds p * (_kdot(vv, dos, c) - deltas[c])
            dkv = _kaccum(dkv, ds * scale, qs, c)
        end
        t += Int32(KT)
    end

    if active
        _kstore!(dk, live ? dkv : ntuple(Returns(zero(Float32)), Val(DIM)), one(Float32), kb, T)
    end
end

"""
    flash_bwd_dv!(dv, q, k, bias, mask, dO, lse,
        scale, n, nouter, nh, mheads, Val(T), Val(DIM), Val(TG), Val(KT), Val(HB), Val(HM))

Backward pass, `dV`, one thread per KEY, streaming query tiles, the same geometry and masking
rules as [`flash_bwd_dk!`](@ref).

`dV[:, kk] = Σ_j p_{j,kk} dO_j` needs only the softmax weight `p`, never `dp` and never `delta`.
So this kernel takes NEITHER `v` NOR `delta` as an argument, they are absent from the signature
rather than merely unused, so nothing binds a buffer the kernel never reads, and it holds only
`kv` plus the `dvv` accumulator: `2*DIM` registers, the same footprint as `flash_fwd!`.

That is what the split buys, and it is why `dV` rather than `dK` is the half worth separating; see
the register-pressure note in the file header. The `s`/`p` recomputed here duplicates
[`flash_bwd_dk!`](@ref)'s, which is pure ALU and repaid many times over by not spilling.
"""
@kernel unsafe_indices = true function flash_bwd_dv!(dv, q, k, bias, mask, dO, lse,
    scale, n, nouter, nh, mheads, ::Val{T}, ::Val{DIM}, ::Val{TG}, ::Val{KT}, ::Val{HB},
    ::Val{HM}) where {T,DIM,TG,KT,HB,HM}
    qs = @localmem T (DIM, KT)
    dos = @localmem T (DIM, KT)
    lses = @localmem Float32 (KT,)

    tid = _i32(@index(Local, Linear))

    # (query or key tile, outer index, batch * head), 0-based, decoded by hand from the linear group index
    ktile, i, bh = _group_ids(_i32(@index(Group, Linear)) - Int32(1), n, nouter, Val(TG))
    kk = ktile * Int32(TG) + tid - Int32(1)

    active = kk < n
    kks = active ? kk : Int32(0)

    hi = _srem(bh, nh)
    bi = _sdiv(bh, nh)
    nn = n * n
    row = (bi * nouter + i) * n
    hstep = nh * Int32(DIM)
    kb = (row + kks) * hstep + hi * Int32(DIM)
    qb0 = row * hstep + hi * Int32(DIM)
    mb = (_sdiv(bh, mheads) * nouter + i) * n     # see `flash_fwd!` for why this is not `bi * nn`
    lrow = (bi * nouter + i) * nh + hi

    masked = active ? (HM ? unsafe_load(mask, mb + kks + Int32(1)) : false) : true
    live = active & !masked

    kv = _kscale(_kload(k, kb, Val(DIM)), scale * _LOG2E)
    dvv = ntuple(Returns(zero(Float32)), Val(DIM))

    # A FULLY MASKED slice (`flash_fwd!` wrote `lse = -Inf` for every query of it) attended uniformly, so
    # `dV[kk] = Σⱼ dO[j] / n` for EVERY key, masked or not. One `lse` load tells, the same for every thread
    # of the group, and the branch is taken once, outside the tile loop: a live slice runs the loop below
    # unchanged. (Only the whole slice is dead this way; a single query row made all-`-Inf` by its bias is not.)
    dead_slice = unsafe_load(lse, lrow * n + Int32(1)) == -Inf32

    t = Int32(0)
    if dead_slice
        invn = inv(Float32(n))
        while t < n
            @synchronize
            _kstage_rows!(qs, dos, q, dO, qb0, hstep, t, n, tid, Val(T), Val(DIM), Val(TG), Val(KT))
            @synchronize
            for c in Int32(1):Int32(KT)
                dvv = _kaccum(dvv, invn, dos, c)
            end
            t += Int32(KT)
        end
    else
        while t < n
            @synchronize
            _kstage_rows!(qs, dos, q, dO, qb0, hstep, t, n, tid, Val(T), Val(DIM), Val(TG), Val(KT))
            _kstage_stats!(lses, nothing, lse, nothing, lrow * n, t, n, tid, Val(TG), Val(KT))
            @synchronize

            for c in Int32(1):Int32(KT)
                jj = min(t + c - Int32(1), n - Int32(1))
                s2 = _kdot(kv, qs, c) + _kbias(bias, bh * nn + jj * n + kks + Int32(1), Val(HB))
                p = @inbounds exp2(s2 - lses[c])
                dvv = _kaccum(dvv, p, dos, c)
            end
            t += Int32(KT)
        end
    end

    if active
        _kstore!(dv, (live | dead_slice) ? dvv : ntuple(Returns(zero(Float32)), Val(DIM)), one(Float32), kb, T)
    end
end

"""
    _kbias(bias, idx, Val(HB)) -> Float32

One bias value, base 2 (`* log2e`), or `0` without reading `bias` when `HB = false`.
"""
@inline _kbias(bias, idx, ::Val{true}) = Float32(unsafe_load(bias, idx)) * _LOG2E
@inline _kbias(bias, idx, ::Val{false}) = zero(Float32)

"""
    _kstage_rows!(qs, dos, q, dO, qb0, hstep, t, n, tid, Val(T), Val(DIM), Val(TG), Val(KT))

Stage one tile of `KT` query rows of `q` and `dO` into local memory, for the key-side kernels.
Rows past `n` are zero-filled; their staged `lse` (see [`_kstage_stats!`](@ref)) kills them.
"""
@inline function _kstage_rows!(qs, dos, q, dO, qb0, hstep, t, n, tid,
    ::Val{T}, ::Val{DIM}, ::Val{TG}, ::Val{KT}) where {T,DIM,TG,KT}
    e = tid
    while e <= Int32(KT) * Int32(DIM)
        d0 = (e - Int32(1)) % Int32(DIM)
        c0 = (e - Int32(1)) ÷ Int32(DIM)
        jj = t + c0
        vq = jj < n ? unsafe_load(q, qb0 + jj * hstep + d0 + Int32(1)) : zero(T)
        vdo = jj < n ? unsafe_load(dO, qb0 + jj * hstep + d0 + Int32(1)) : zero(T)
        @inbounds qs[d0+Int32(1), c0+Int32(1)] = vq
        @inbounds dos[d0+Int32(1), c0+Int32(1)] = vdo
        e += Int32(TG)
    end
    return nothing
end

"""
    _kstage_stats!(lses, deltas, lse, delta, lbase, t, n, tid, Val(TG), Val(KT))

Stage one query tile's `lse` (converted to base 2) and, unless `deltas === nothing`, `delta`. A query
past `n` gets `lse = +floatmax`, so its `p = exp2(s - floatmax)` is exactly zero.
"""
@inline function _kstage_stats!(lses, deltas, lse, delta, lbase, t, n, tid,
    ::Val{TG}, ::Val{KT}) where {TG,KT}
    e = tid
    while e <= Int32(KT)
        jj = t + e - Int32(1)
        ok = jj < n
        jl = ok ? jj : Int32(0)
        @inbounds lses[e] = ok ? unsafe_load(lse, lbase + jl + Int32(1)) * _LOG2E : floatmax(Float32)
        if deltas !== nothing
            @inbounds deltas[e] = ok ? unsafe_load(delta, lbase + jl + Int32(1)) : zero(Float32)
        end
        e += Int32(TG)
    end
    return nothing
end

"""
    flash_bwd_delta(o, dO) -> Array{Float32}

`delta[j,i,bh] = Σ_d o[d,j,i,bh] * dO[d,j,i,bh]`, shape matching `lse`, the row constant both
backward kernels need (`Σ_j p_{j,kk}(dP_{j,kk} - delta_j) = 0` is the softmax-Jacobian identity that
makes `dS = p * (dP - delta)` correct; see e.g. the FlashAttention-2 paper).

Ordinary broadcasting, not a `@kernel`: the RESULT costs nothing extra to materialise (it is exactly
the shape of `o`/`dO` reduced over one axis already the same size as `lse`), so a raw-pointer kernel
buys nothing here. Works on any array backend — CPU, `MtlArray`, whatever `q`/`k`/`v` already are,
for the same reason `SDPA()`'s CPU fallback needs no device-specific kernel of its own either,
see `src/kernels/sdpa_op.jl`.

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
`Metal.atomic_fetch_add_explicit`: ordinary dispatch on a function this package owns, like the
backend hooks in `device.jl`.
"""
function _atomic_fadd! end
