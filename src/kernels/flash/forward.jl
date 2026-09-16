# A FlashAttention style kernel for memory efficient triangle attention, written in pure Julia & KernelAbstractions.
#
# This file is backend-agnostic on purpose: no Metal, CUDA or ROCm symbol appears in it. A backend
# extension supplies only the things KernelAbstractions cannot — ahead-of-time compilation to that backend's binary
# format, the device's occupancy limits, and the launch itself. Everything about the ALGORITHM,
# including the launch geometry, lives here.
#
# Three constraints shape the code below:
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
#   * `m` and `corr = exp(m - mn)` ARE the numerical-stability mechanism of online softmax;
#   * `accv` accumulates over `n` terms with repeated rescaling.
#
# `lse` is likewise `Float32`: it is the residual the backward reconstructs probabilities from.
# So low-precision storage buys bandwidth and local-memory headroom (a smaller `T` makes larger `KT`
# legal, see `candidate_configs`) without touching the arithmetic that needs the range.

"""
    flash_fwd!(o, lse, q, k, v, bias, mask, scale, neg_inf, n, nouter, nh, mheads,
        Val(T), Val(DIM), Val(TG), Val(KT))

Fused attention forward: online softmax over key tiles, one thread per query.

`unsafe_indices=true` drops KernelAbstractions' implicit `__validindex` guard. That is deliberate and safe: the
kernel does its own bounds handling via `active = j < n`, and dropping the guard is what allows the
sequence length to be a runtime argument rather than baked into the compiled code.

Threads past the end must NOT return early, they still take part in the cooperative tile load and,
critically, in the barriers. They are given a clamped index so every address stays in bounds and
their result is simply never stored. This is what makes the kernel correct at ANY `n`, with no
padding to a multiple of the tile size.

# Layout
`n` is the attended length (queries = keys) and `nouter` the OUTER length the attention is batched over.
They are equal for triangle attention (`[C, N, N, B]`), and differ for everything else: MSA row
attention attends over `n = N_res` for each of `nouter = N_seq` rows, column attention the reverse, and a
plain `[C, N, B]` sequence has `nouter = 1`.

`q`, `k`, `v`, `o` are `[DIM, nh, n, nouter, B]` with `DIM` fastest and element type `T`, where `nh` is
the number of heads interleaved between the channel and query axes: `nh = 1` is the packed
`[DIM, n, nouter, bh]` case. `bias` is `[n, n, bh]` of `T` — keys fastest, and shared by every outer
index, which is exactly a pair bias; `lse` is `[n, nh, nouter, B]` of `Float32` (`[n, nouter, bh]` when
`nh = 1`). `mask` is `[n, nouter, bh ÷ mheads]` of `Bool` (a KEY mask per outer index), see `mheads`
below; the common case is one mask per batch element, `[n, nouter, B]`. All raw pointers; see the note
at the top of this file. See the addressing comment in the body for why `nh` is a runtime argument.

# Precision
`T` is storage only. `scale`, `neg_inf`, every accumulator and `lse` are `Float32`, see note 3 at
the top of this file for why that is a requirement and not a default.
"""
@kernel unsafe_indices = true function flash_fwd!(o, lse, q, k, v, bias, mask,
    scale, neg_inf, n, nouter, nh, mheads, ::Val{T}, ::Val{DIM}, ::Val{TG}, ::Val{KT}) where {T,DIM,TG,KT}
    ks = @localmem T (DIM, KT)
    vs = @localmem T (DIM, KT)

    gx, gy, gz = @index(Group, NTuple)
    tid = Int32(@index(Local, Linear))

    jt = Int32(gx) - Int32(1)       # query tile
    i = Int32(gy) - Int32(1)        # outer index in `0:nouter-1`, one per group, by construction
    bh = Int32(gz) - Int32(1)       # batch * head
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
    hi = bh % nh                      # head index within its batch element
    bi = bh ÷ nh                      # batch element
    nn = n * n
    row = (bi * nouter + i) * n       # first query slot of this (batch, i) row
    hstep = nh * Int32(DIM)           # elements from one query slot to the next
    qb = (row + js) * hstep + hi * Int32(DIM)
    kb = row * hstep + hi * Int32(DIM)
    bb = bh * nn + js * n
    # `mheads` is a SEPARATE stride from `nh`: a mask never varies by head (LTA's own mask has no
    # head axis at all), but it CAN vary by batch, and `bh ÷ nh` is the wrong batch index whenever
    # `nh` is not the true head count, which is exactly the packed `nh = 1` case, where `bh`
    # conflates heads and batch with no way to recover the split from `nh` alone. `mheads` is that
    # true head count, supplied separately, so `bh ÷ mheads` is always the real batch index.
    mb = ((bh ÷ mheads) * nouter + i) * n
    lrow = (bi * nouter + i) * nh + hi # `lse` has no channel axis, so it strides differently

    qv = _kload(q, qb, Val(DIM))
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
            # Ragged final key tile: load zeros rather than reading past the end. The compute loop
            # skips these columns, so only the address matters, never the value.
            vk = kk < n ? unsafe_load(k, kb + kk * hstep + d0 + Int32(1)) : zero(T)
            vv = kk < n ? unsafe_load(v, kb + kk * hstep + d0 + Int32(1)) : zero(T)
            @inbounds ks[d0+Int32(1), c0+Int32(1)] = vk
            @inbounds vs[d0+Int32(1), c0+Int32(1)] = vv
            e += Int32(TG)
        end
        @synchronize

        for c in Int32(1):Int32(KT)
            kk = t + c - Int32(1)
            if kk < n
                @inbounds masked = unsafe_load(mask, mb + kk + Int32(1))
                if !masked
                    s = _kdot(qv, ks, c) * scale +
                        Float32(unsafe_load(bias, bb + kk + Int32(1)))
                    mn = max(m, s)
                    corr = exp(m - mn)
                    p = exp(s - mn)
                    l = l * corr + p
                    accv = _kupd(accv, corr, p, vs, c)
                    m = mn
                end
            end
        end
        t += Int32(KT)
    end

    if active
        _kstore!(o, accv, inv(l), qb, T)
        unsafe_store!(lse, m + log(l), lrow * n + j + Int32(1))
    end
end


"""
    _kdot(qv, ks, c) -> Float32

Inner product of a register-resident query against column `c` of the staged key tile.

Written as a loop rather than `sum(ntuple(d -> qv[d] * ks[d, c], Val(D)))` because **a closure that
captures a kernel argument loses its type under `@kernel`** — inference gives up and compilation
fails with "unsupported call to jl_mul_float". Register residency matters for `qv` and `accv`, not
for traversing `ks`, so a loop costs nothing here.
"""
@inline function _kdot(qv::NTuple{D,Float32}, ks, c) where {D}
    acc = zero(Float32)
    for d in 1:D
        @inbounds acc += qv[d] * Float32(ks[d, c])
    end
    return acc
end

"""
    _kload(q, qb, Val(D)) -> NTuple{D,Float32}

Load one query vector into registers, widening to the compute precision. Takes the pointer and
offset as PARAMETERS for the reason given in [`_kdot`](@ref): as a captured closure variable the
pointer goes dynamic and compilation fails with "unsupported dynamic function invocation (call to
pointerref)".
"""
@inline _kload(q, qb, ::Val{D}) where {D} =
    ntuple(d -> Float32(unsafe_load(q, qb + Int32(d))), Val(D))

"""Online-softmax rescale of the accumulator. `accv` is a parameter, not a captured local."""
@inline _kupd(accv::NTuple{D,Float32}, corr, p, vs, c) where {D} =
    ntuple(d -> (@inbounds accv[d] * corr + p * Float32(vs[d, c])), Val(D))

"""Narrow the Float32 accumulator back to the storage type on the way out."""
@inline function _kstore!(o, accv::NTuple{D,Float32}, inv_l, qb, ::Type{T}) where {D,T}
    ntuple(Val(D)) do d
        @inbounds unsafe_store!(o, T(accv[d] * inv_l), qb + Int32(d))
        nothing
    end
    return nothing
end

"""
    launch_geometry(n, heads, tg; nouter=n) -> (; ndrange, workgroupsize, ngroups)

The launch parameters for a problem, in the 1-D form KernelAbstractions requires (see the note at the top).

`ndrange` is expressed in THREADS and is what the context's iterspace is built from; `ngroups` is
the number of groups a backend should actually dispatch, as a 1-D grid. `nouter` is the outer
extent (see [`flash_fwd!`](@ref)); it defaults to `n`, the triangle-attention case.
"""
function launch_geometry(n::Integer, heads::Integer, tg::Integer; nouter::Integer=n)
    ndrange = (cld(n, tg) * tg, nouter, heads)
    return (; ndrange, workgroupsize=(tg, 1, 1), ngroups=cld(n, tg) * nouter * heads)
end

"""
    kernel_object(backend, tg; ndrange=nothing) -> KernelAbstractions.Kernel

The KernelAbstractions kernel specialised for a backend and group size.

Passing `ndrange` bakes the problem size in, which makes the context a **ghost type** — it occupies
no buffer slot, giving the cleanest possible ABI, at the cost of one compiled binary per `n`.
Leaving it `nothing` keeps the context a real 16-byte argument that carries the iterspace, and the
compiled binary is then **independent of `n`**.

Measured on Metal: the two are within 1% of each other at every size, i.e. inside noise. Prefer the
`n`-independent form — the binary is reusable and nothing is given up for it.
"""
kernel_object(backend, tg::Integer; ndrange=nothing) =
    ndrange === nothing ? flash_fwd!(backend, (tg, 1, 1)) :
    flash_fwd!(backend, (tg, 1, 1), ndrange)

"""
    kernel_context(backend, n, heads, tg; nouter=n, static=false) -> (ctx, ngroups, is_ghost)

Build the KernelAbstractions launch context for a problem.

The context is produced by `KernelAbstractions.mkcontext`, never by hand: a backend that needs it in a buffer should
upload this value as-is. Its internal layout is KernelAbstractions's business and must not be assumed.

`nouter` changes only the iterspace the context CARRIES, never the compiled code, so one binary
serves every `(n, nouter)` — except under `static`, which bakes the whole ndrange in.
"""
function kernel_context(backend, n::Integer, heads::Integer, tg::Integer;
    nouter::Integer=n, static::Bool=false)
    g = launch_geometry(n, heads, tg; nouter)
    obj = kernel_object(backend, tg; ndrange=(static ? g.ndrange : nothing))
    nd, _, iterspace, _ = KernelAbstractions.launch_config(obj, g.ndrange, g.workgroupsize)
    ctx = KernelAbstractions.mkcontext(obj, nd, iterspace)
    return (ctx, length(KernelAbstractions.blocks(iterspace)), sizeof(typeof(ctx)) == 0)
end

"""
    kernel_signature(ctx_type, PT, PL, PB, T, dim, tg, kt) -> Type

Argument tuple type for ahead-of-time compilation. Backends hand this to their own compiler; the
ordering here is the contract the `stablehlo.custom_call` binds against.

`PT` is a pointer to the storage type `T` (used for `q`, `k`, `v`, `o`, `bias`), `PL` a pointer to
`Float32` (the `lse` residual, which is never narrowed), and `PB` a pointer to `Bool` (the mask).
The four `Int32`s are `n`, `nouter`, `nh`, `mheads`.
"""
kernel_signature(::Type{C}, ::Type{PT}, ::Type{PL}, ::Type{PB},
    ::Type{T}, dim, tg, kt) where {C,PT,PL,PB,T} =
    Tuple{C,PT,PL,PT,PT,PT,PT,PB,Float32,Float32,Int32,Int32,Int32,Int32,
        Val{T},Val{dim},Val{tg},Val{kt}}
