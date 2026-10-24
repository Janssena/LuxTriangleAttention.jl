# A FlashAttention style kernel for memory efficient triangle attention, written in pure Julia & KernelAbstractions.
#
# This file is backend-agnostic on purpose: no Metal, CUDA or ROCm symbol appears in it. A backend
# extension supplies only the things KernelAbstractions cannot — ahead-of-time compilation to that backend's binary
# format, the device's occupancy limits, and the launch itself. Everything about the ALGORITHM,
# including the launch geometry, lives here.
#
# Four constraints shape the code below:
#
# ## 1. The dispatch must be a 1-D grid of all groups
#
# `Metal/src/MetalKernels.jl` implements the group index as
#
#     __index_Group_Linear(ctx) = threadgroup_position_in_grid().x
#     __index_Group_NTuple(ctx) = blocks(__iterspace(ctx))[threadgroup_position_in_grid().x]
#
# it reads ONLY the x component and rebuilds N-D indices in software from the context's
# iterspace. A native 3-D threadgroup grid is therefore inexpressible: y and z are silently
# dropped, and the kernel writes exactly one (i, bh) slice while the rest of the output stays
# untouched. So the grid is 1-D over `cld(N,TG) * N * heads` groups and `@index(Group, NTuple)`
# does the decode. Check this first when bringing up another backend.
#
# ## 2. Data is passed as raw pointers, not device arrays
#
# An `AbstractGPUArray` argument is passed to the kernel as a STRUCT (pointer plus dimensions) in
# its buffer. A launcher that binds raw data buffers, which is what the `stablehlo.custom_call`
# path does, therefore binds it wrong, and the kernel silently writes nothing. With raw pointers
# the emitted ABI is byte-identical to a hand-written kernel's and the custom call binds unchanged.
#
# The consequence: KernelAbstractions is used here for the parallel STRUCTURE — indices, local memory, barriers,
# launch geometry — and not for data access. That is still the whole of what differs per backend.
#
# ## 3. Two precisions, not one
#
# `T` is the STORAGE type of `q`, `k`, `v`, `o` and `bias`. Accumulation is **always `Float32`**, and
# that is a correctness requirement. Three values would be ruined by low precision:
#
#   * `l`, the softmax denominator, is a sum over all `n` keys and divides every output element;
#   * `m` and `corr = exp2(m - mn)` ARE the numerical-stability mechanism of online softmax;
#   * `accv` accumulates over `n` terms with repeated rescaling.
#
# `lse` is likewise `Float32`: it is the residual the backward reconstructs probabilities from.
# So low-precision storage buys bandwidth and local-memory headroom (a smaller `T` makes larger `KT`
# legal, see `admissible` in `flash/tune.jl`) without touching the arithmetic that needs the range.
#

const _LOG2E = Float32(1.4426950408889634)
const _LN2 = Float32(0.6931471805599453)

# `Int32(x)` of a runtime `Int64` (a KernelAbstractions index) can throw an `InexactError`, and a device binary built
# without a runtime (KernelDispatch's CUDA binaries) is refused for any code that can throw. Group and thread
# indices of one launch always fit, so the unchecked truncation is exact. Use it for every runtime conversion in
# a kernel; a conversion of a compile-time constant folds away and may stay `Int32(...)`.
@inline _i32(x::Integer) = unsafe_trunc(Int32, x)

# `÷` and `%` by a RUNTIME divisor check for zero (a `DivideError`) and refuse the same builds. The divisors in the
# kernels (`nh`, `mheads`, `nouter`, the tile counts) are positive launch parameters, and the dividends are
# non-negative indices, so the plain signed division and remainder give the same results.
@inline _sdiv(a::T, b::T) where {T<:Signed} = Core.Intrinsics.sdiv_int(a, b)
@inline _srem(a::T, b::T) where {T<:Signed} = Core.Intrinsics.srem_int(a, b)

"""
    flash_fwd!(o, lse, q, k, v, bias_t, mask, scale, neg_inf, n, nouter, nh, mheads,
        Val(T), Val(DIM), Val(TG), Val(KT), Val(SB), Val(HB), Val(HM))

Fused attention forward: online softmax over key tiles, one thread per query.

`unsafe_indices=true` drops KernelAbstractions' implicit `__validindex` guard. That is deliberate and safe: the
kernel does its own bounds handling via `active = j < n`, and dropping the guard is what allows the
sequence length to be a runtime argument rather than baked into the compiled code.

Threads past the end must NOT return early, they still take part in the cooperative tile load and,
critically, in the barriers. They are given a clamped index so every address stays in bounds and
their result is simply never stored. This is what makes the kernel correct at ANY `n`, with no
padding to a multiple of the tile size.

# Algorithm
Per key tile of `KT` keys, the group stages `k`, `v` and the tile's key mask into local memory; each
thread then walks the tile in SUB-BLOCKS of `SB` keys. A sub-block's `SB` scores are formed first,
their maximum updates the running maximum ONCE, and the accumulator is rescaled once — rather than
once per key, which costs a full `DIM`-wide multiply and a second exponential per key. `SB` trades
that saving against `SB` extra live registers and is tuned per device (see `flash/tune.jl`).

Softmax runs in base 2: `scale·log2(e)` is folded into `q` once and `log2(e)` into each bias value,
so every exponential is a bare `exp2`. `lse` is converted back to the natural log on the way out.

The key mask is staged as an ADDITIVE term, `0` for a live key and `neg_inf` (a large FINITE
negative, `-floatmax`) for a masked one or one past `n`, so the inner loop has no branch. A masked
key then has `p = exp2(≈ -floatmax - m) = 0` exactly wherever its row has a live key. A row whose
leading keys are all masked accumulates finite garbage until its first live key arrives, and that
key's correction factor `exp2(≈ -floatmax - m_live) = 0` wipes it exactly. A row with NO live key
never leaves `m ≈ neg_inf`: every score is `neg_inf` exactly (the real score is absorbed, `|s| ≪ ulp(floatmax)`),
so every key, live or past `n`, has `p = 1`, and the accumulator holds `Σₖ vₖ` (keys past `n` stage `v = 0`).
It is detected at the end and written as UNIFORM attention, `o = Σₖ vₖ / n` (the mean of `v`, what OpenFold's and
Boltz2's finite `-1e9` mask gives such a row; LuxLib's `-Inf` would give NaN) and `lse = -Inf`, the marker
the backward reads. The branch is after the key loop, so the hot loop is the same code as without it.

# Layout
`n` is the attended length (queries = keys) and `nouter` the OUTER length the attention is batched over.
They are equal for triangle attention (`[C, N, N, B]`), and differ for everything else: MSA row
attention attends over `n = N_res` for each of `nouter = N_seq` rows, column attention the reverse, and a
plain `[C, N, B]` sequence has `nouter = 1`.

`q`, `k`, `v`, `o` are `[DIM, nh, n, nouter, B]` with `DIM` fastest and element type `T`, where `nh` is
the number of heads interleaved between the channel and query axes: `nh = 1` is the packed
`[DIM, n, nouter, bh]` case. `bias_t` is the pair bias TRANSPOSED, `[n (queries), n (keys), bh]` of
`T` — QUERIES fastest, the registry's `:bias_t` role — so that the 32 threads of a SIMD group, which
own 32 consecutive queries, read 32 consecutive elements for each key. In the keys-fastest `:bias`
layout the same read touches 32 different cache lines; measured on an M1 Pro that single change was
worth 1.3x of the forward. `lse` is `[n, nh, nouter, B]` of `Float32` (`[n, nouter, bh]` when
`nh = 1`). `mask` is `[n, nouter, bh ÷ mheads]` of `Bool` (a KEY mask per outer index, `true` =
DROP), see `mheads` below; the common case is one mask per batch element, `[n, nouter, B]`. All raw
pointers; see the note at the top of this file.

# Precision
`T` is storage only. `scale`, `neg_inf`, every accumulator and `lse` are `Float32`, see note 3 at
the top of this file for why that is a requirement and not a default.

# Absent bias or mask
`HB` (has bias) and `HM` (has mask) are compile-time flags. `HB = false` drops the bias term from
every score — one global load per (query, key), `H·N³` over a triangle-attention call — and `HM =
false` stages every in-range key as live without reading `mask`. The corresponding pointer is then
never dereferenced, so a launcher binds any placeholder buffer: the ABI, and so the registry's
buffer table, is the same for all four specialisations. `HB = HM = true` is the general kernel.
"""
@kernel unsafe_indices = true function flash_fwd!(o, lse, q, k, v, bias_t, mask,
    scale, neg_inf, n, nouter, nh, mheads,
    ::Val{T}, ::Val{DIM}, ::Val{TG}, ::Val{KT}, ::Val{SB}, ::Val{HB},
    ::Val{HM}) where {T,DIM,TG,KT,SB,HB,HM}
    ks = @localmem T (DIM, KT)
    vs = @localmem T (DIM, KT)
    ms = @localmem Float32 (KT,)

    gx, gy, gz = @index(Group, NTuple)
    tid = _i32(@index(Local, Linear))

    jt = _i32((gx)) - Int32(1)       # query tile
    i = _i32((gy)) - Int32(1)        # outer index in `0:nouter-1`, one per group, by construction
    bh = _i32((gz)) - Int32(1)       # batch * head
    j = jt * Int32(TG) + tid - Int32(1)

    active = j < n
    js = active ? j : Int32(0)

    # `nh` is the stride, in query slots, from one head to the next, the one runtime number that
    # lets a single binary serve both layouts this package feeds it. `nh = 1` is the packed
    # `[DIM, n, n, bh]` buffers the direct-Metal path builds, where each `(i, bh)` slice is
    # contiguous. `nh = H` is LuxTriangleAttention's own `[hd, H, Nq, Nj, B]`, where the head axis
    # sits between the channel and query axes. `DIM` stays the fastest axis either way, so the inner
    # product still walks `DIM` contiguous elements and only the step between queries (or keys)
    # grows. Without this the traced path had to materialise packed copies of `q`, `k` and `v` and
    # transpose `o` back: four full-size buffers, which doubled the XLA path's device memory over
    # the direct one.
    hi = _srem(bh, nh)                      # head index within its batch element
    bi = _sdiv(bh, nh)                      # batch element
    nn = n * n
    row = (bi * nouter + i) * n       # first query slot of this (batch, i) row
    hstep = nh * Int32(DIM)           # elements from one query slot to the next
    qb = (row + js) * hstep + hi * Int32(DIM)
    kb = row * hstep + hi * Int32(DIM)
    btb = bh * nn + js                # bias_t: (query js, key kk) lives at btb + kk * n
    # `mheads` is a SEPARATE stride from `nh`: a mask never varies by head (LTA's own mask has no
    # head axis at all), but it CAN vary by batch, and `bh ÷ nh` is the wrong batch index whenever
    # `nh` is not the true head count, which is exactly the packed `nh = 1` case, where `bh`
    # conflates heads and batch with no way to recover the split from `nh` alone. `mheads` is that
    # true head count, supplied separately, so `bh ÷ mheads` is always the real batch index.
    mb = ((_sdiv(bh, mheads)) * nouter + i) * n
    lrow = (bi * nouter + i) * nh + hi # `lse` has no channel axis, so it strides differently

    qv = _kscale(_kload(q, qb, Val(DIM)), scale * _LOG2E)
    accv = ntuple(Returns(zero(Float32)), Val(DIM))
    m = neg_inf
    l = zero(Float32)

    t = Int32(0)
    while t < n
        @synchronize
        e = tid
        while e <= Int32(KT) * Int32(DIM)
            d0 = (e - Int32(1)) % Int32(DIM)
            c0 = (e - Int32(1)) ÷ Int32(DIM)
            kk = t + c0
            # Ragged final key tile: load zeros rather than reading past the end. The staged mask
            # below kills these columns, so only the address matters, never the value.
            vk = kk < n ? unsafe_load(k, kb + kk * hstep + d0 + Int32(1)) : zero(T)
            vv = kk < n ? unsafe_load(v, kb + kk * hstep + d0 + Int32(1)) : zero(T)
            @inbounds ks[d0+Int32(1), c0+Int32(1)] = vk
            @inbounds vs[d0+Int32(1), c0+Int32(1)] = vv
            e += Int32(TG)
        end
        # The tile's key mask as an additive term (see "Algorithm" above).
        _kstage_mask!(ms, mask, mb, t, n, tid, neg_inf, Val(TG), Val(KT), Val(HM))
        @synchronize

        c0 = Int32(0)
        while c0 < Int32(KT)
            s = _kscores(qv, ks, bias_t, btb, t, c0, ms, n, Val(SB), Val(HB))
            mn = max(m, maximum(s))
            corr = exp2(m - mn)
            p = _kexp2(s, mn)
            l = l * corr + sum(p)
            accv = _kupdate(accv, corr, p, vs, c0)
            m = mn
            c0 += Int32(SB)
        end
        t += Int32(KT)
    end

    if active
        if m <= 0.5f0 * neg_inf    # no live key in this row: see "Algorithm" above
            _kstore!(o, accv, inv(Float32(n)), qb, T)
            unsafe_store!(lse, -Inf32, lrow * n + j + Int32(1))
        else
            _kstore!(o, accv, inv(l), qb, T)
            unsafe_store!(lse, (m + log2(l)) * _LN2, lrow * n + j + Int32(1))
        end
    end
end

# ---------------------------------------------------------------------------------------------
# Per-channel helpers. Every one that touches a `DIM`-wide register tuple is straight-line code
# generated for the exact `DIM`, with an inline hint — see note 4 at the top of this file for why a
# loop, or an un-inlined call, is not acceptable here.
# ---------------------------------------------------------------------------------------------

"""
    _kdot(qv, ks, c) -> Float32

Inner product of a register-resident vector against column `c` of a staged tile, fully unrolled.

Also the reason every helper here takes its pointers and offsets as PARAMETERS: **a closure that
captures a kernel argument loses its type under `@kernel`** — inference gives up and compilation
fails with "unsupported call to jl_mul_float".
"""
@generated function _kdot(qv::NTuple{D,Float32}, ks, c) where {D}
    ex = :(qv[1] * Float32(@inbounds ks[1, c]))
    for d in 2:D
        ex = :(muladd(qv[$d], Float32(@inbounds ks[$d, c]), $ex))
    end
    return Expr(:block, Expr(:meta, :inline), ex)
end

"""
    _kscores(qv, ks, bias_t, btb, t, c0, ms, n, Val(SB), Val(HB)) -> NTuple{SB,Float32}

Base-2 scores of the `SB` keys `c0+1 … c0+SB` of the staged tile: `q·k` (with `scale·log2e`
already in `q`) plus, when `HB`, the transposed bias times `log2e`, plus the staged additive mask. A
key past `n` is clamped to a valid bias address; its mask term already kills it. With `HB = false`
the bias term is not generated at all, so `bias_t` is never read.
"""
@generated function _kscores(qv::NTuple{D,Float32}, ks, bias_t, btb, t, c0, ms, n,
    ::Val{SB}, ::Val{HB}) where {D,SB,HB}
    body = Expr[]
    for u in 1:SB
        su = Symbol(:s, u)
        score = HB ?
            :(_kdot(qv, ks, c) + Float32(unsafe_load(bias_t, btb + kk * n + Int32(1))) * _LOG2E + ms[c]) :
            :(_kdot(qv, ks, c) + ms[c])
        push!(body, quote
            c = c0 + Int32($u)
            kk = min(t + c - Int32(1), n - Int32(1))
            $su = @inbounds $score
        end)
    end
    return quote
        $(Expr(:meta, :inline))
        $(body...)
        $(Expr(:tuple, (Symbol(:s, u) for u in 1:SB)...))
    end
end

"""
    _kstage_mask!(ms, mask, mb, t, n, tid, dead_value, Val(TG), Val(KT), Val(HM))

Stage one key tile's mask into local memory as an additive term: `0` for a live key, `dead_value`
for a masked key or one past `n`. Strided by `TG`, so it is correct for `KT > TG` too. With
`HM = false` only the keys past `n` are dead and `mask` is never read.
"""
@inline function _kstage_mask!(ms, mask, mb, t, n, tid, dead_value, ::Val{TG}, ::Val{KT},
    ::Val{HM}) where {TG,KT,HM}
    e = tid
    while e <= Int32(KT)
        kk = t + e - Int32(1)
        dead = kk < n ? (HM ? unsafe_load(mask, mb + kk + Int32(1)) : false) : true
        @inbounds ms[e] = dead ? dead_value : zero(Float32)
        e += Int32(TG)
    end
    return nothing
end

"""`exp2.(s .- mn)`, straight-line."""
@generated _kexp2(s::NTuple{SB,Float32}, mn) where {SB} =
    Expr(:block, Expr(:meta, :inline), Expr(:tuple, (:(exp2(s[$u] - mn)) for u in 1:SB)...))

"""
    _kupdate(accv, corr, p, vs, c0) -> NTuple{D,Float32}

The online-softmax accumulator update for one sub-block: `accv .* corr .+ Σ_u p[u] .* vs[:, c0+u]`,
one rescale for all `SB` keys.
"""
@generated function _kupdate(accv::NTuple{D,Float32}, corr, p::NTuple{SB,Float32}, vs,
    c0) where {D,SB}
    elems = map(1:D) do d
        ex = :(accv[$d] * corr)
        for u in 1:SB
            ex = :(muladd(p[$u], Float32(@inbounds vs[$d, c0+Int32($u)]), $ex))
        end
        ex
    end
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, elems...))
end

"""`accv .+ w .* ks[:, c]` — a plain accumulate, no rescale (the backward carries no running max)."""
@generated _kaccum(accv::NTuple{D,Float32}, w, ks, c) where {D} =
    Expr(:block, Expr(:meta, :inline),
        Expr(:tuple, (:(muladd(w, Float32(@inbounds ks[$d, c]), accv[$d])) for d in 1:D)...))

"""`accv .* w`"""
@generated _kscale(accv::NTuple{D,Float32}, w) where {D} =
    Expr(:block, Expr(:meta, :inline), Expr(:tuple, (:(accv[$d] * w) for d in 1:D)...))

"""
    _kload(q, qb, Val(D)) -> NTuple{D,Float32}

Load one vector into registers, widening to the compute precision. Takes the pointer and offset as
PARAMETERS for the reason given in [`_kdot`](@ref).
"""
@generated _kload(q, qb, ::Val{D}) where {D} =
    Expr(:block, Expr(:meta, :inline),
        Expr(:tuple, (:(Float32(unsafe_load(q, qb + Int32($d)))) for d in 1:D)...))

"""
    _kstore!(o, accv, w, qb, T)

Store `accv .* w`, narrowed to the storage type. `w = inv(l)` in the forward, `1` or `0` in the
backward.
"""
@generated function _kstore!(o, accv::NTuple{D,Float32}, w, qb, ::Type{T}) where {D,T}
    stores = [:(unsafe_store!(o, T(accv[$d] * w), qb + Int32($d))) for d in 1:D]
    return Expr(:block, Expr(:meta, :inline), stores..., :(return nothing))
end

"""
    stage_kernel(stage) -> KernelAbstractions kernel function

The `@kernel` implementing one stage of the built-in entry: `:fwd`, `:dq`, `:dk` or `:dv`.
"""
stage_kernel(stage::Symbol) =
    stage === :fwd ? flash_fwd! :
    stage === :dq ? flash_bwd_dq! :
    stage === :dk ? flash_bwd_dk! :
    stage === :dv ? flash_bwd_dv! :
    throw(ArgumentError("no built-in kernel for stage `:$stage`; stages are $(FLASH_STAGES)"))

"""
    kernel_argtypes(stage, T, dim, cfg, hasbias=true, hasmask=true) -> Tuple

The argument types of one stage's kernel AFTER KernelAbstractions' context, in signature order, for
`KernelDispatch.ka_compile`: `AbstractArray{X}` for a buffer (bound as a raw device pointer to `X`), a bits type
for a scalar, `Val{x}` for a compile-time parameter (no buffer slot, so the ABI is the same for every
`hasbias`/`hasmask` combination). This is the contract the launch binds against: [`SDPA`](@ref)'s
`forward`/`backward` pass exactly these arguments, in this order.

The storage type `T` is used for `q`, `k`, `v`, `o`, `dO` and the bias; `lse`, `delta` and `dbias` are
`Float32`, never narrowed; the mask is `Bool`. The four `Int32`s are `n`, `nouter`, `nh`, `mheads`.
"""
function kernel_argtypes(stage::Symbol, ::Type{T}, dim::Integer, cfg,
    hb::Bool=true, hm::Bool=true) where {T}
    A(x) = AbstractArray{x}
    tg, kt = cfg.tg, cfg.kt
    ints = (Int32, Int32, Int32, Int32)
    flags = (Val{hb}, Val{hm})
    return stage === :fwd ?
           (A(T), A(Float32), A(T), A(T), A(T), A(T), A(Bool), Float32, Float32, ints...,
        Val{T}, Val{dim}, Val{tg}, Val{kt}, Val{cfg.sb}, flags...) :
           stage === :dq ?
           (A(T), A(Float32), A(T), A(T), A(T), A(T), A(Bool), A(T), A(Float32), A(Float32), Float32, ints...,
        Val{T}, Val{dim}, Val{tg}, Val{kt}, flags...) :
           stage === :dk ?
           (A(T), A(T), A(T), A(T), A(T), A(Bool), A(T), A(Float32), A(Float32), Float32, ints...,
        Val{T}, Val{dim}, Val{tg}, Val{kt}, flags...) :
           stage === :dv ?
           (A(T), A(T), A(T), A(T), A(Bool), A(T), A(Float32), Float32, ints...,
        Val{T}, Val{dim}, Val{tg}, Val{kt}, flags...) :
           throw(ArgumentError("no built-in kernel for stage `:$stage`; stages are $(FLASH_STAGES)"))
end
