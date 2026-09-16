# Metal's half of the flash kernel: ahead-of-time compilation to a `.metallib`, and the launch.
#
# The kernel source is not here — it lives once, backend-agnostic, in `src/kernels/flash/forward.jl`
# and `backward.jl`. What Metal must
# supply is everything KernelAbstractions has no notion of: a standalone binary with a known buffer
# ABI (because it is embedded into an XLA graph as a `stablehlo.custom_call`, not launched by KA),
# and the encoder that binds buffers to it.

# Pipeline creation (metallib load -> MTLFunction -> pipeline state) is expensive; leaving it on the
# launch path put a large constant on every measurement and dominated small problems.
const _PIPES = Dict{Tuple{String,String},Any}()
function pipeline(path, entry)
    get!(_PIPES, (path, entry)) do
        dev = Metal.device()
        Metal.MTLComputePipelineState(dev,
            Metal.MTLFunction(Metal.MTLLibraryFromFile(dev, path), entry))
    end
end

# Metal caps live command queues at ~64; one per launch survived a few reps and then threw
# `UndefRefError`, after first producing impossibly fast timings. One queue per process.
const _QUEUE = Ref{Any}(nothing)
queue() = (_QUEUE[] === nothing && (_QUEUE[] = Metal.MTLCommandQueue(Metal.device())); _QUEUE[])

function launch(path, entry, args, ngroups, tg)
    pipe = pipeline(path, entry)
    tg <= Int(pipe.maxTotalThreadsPerThreadgroup) || throw(ArgumentError(
        "TG=$tg exceeds this pipeline's limit of $(pipe.maxTotalThreadsPerThreadgroup); " *
        "dispatching over it returns garbage AND a fast time (see `occupancy_limit`)"))
    keep = Any[]
    cb = Metal.MTLCommandBuffer(queue())
    enc = Metal.MTLComputeCommandEncoder(cb)
    Metal.set_function!(enc, pipe)
    for (i, a) in enumerate(args)
        buf, off = a isa MtlArray ? (a.data[], a.offset * sizeof(eltype(a))) :
                   (begin
            s = MtlArray([a])
            push!(keep, s)
            s.data[]
        end, 0)
        Metal.set_buffer!(enc, buf, off, i)   # 1-based, despite Metal's own 0-based convention
    end
    # A 1-D grid of all groups: KA's Metal backend reads only the x component of the threadgroup
    # position and rebuilds (jt, i, bh) in software. See the shared kernel file.
    Metal.dispatchThreadgroups!(enc, Metal.MTLSize(ngroups, 1, 1), Metal.MTLSize(tg, 1, 1))
    Metal.endEncoding!(enc)
    Metal.commit!(cb)
    Metal.wait_completed(cb)
    return nothing
end

const NEG_INF = -floatmax(Float32)

"""
    run_fwd!(o, lse, q, k, v, bias, mask; tg, kt, n, heads, dim, nouter=n, nh=1, mheads=heads, scale)

Run the forward on device arrays already in the kernel's `[dim, n, nouter, bh]` layout. `nouter`
defaults to `n`, the triangle-attention case (see `flash_fwd!`).

`nh` is the kernel's head stride (see `flash_fwd!`). It defaults to `1`, the packed layout this
entry point documents; `nh = H` addresses LuxTriangleAttention's own `[hd, H, Nq, Nj, B]` instead,
which is what the traced path passes. `heads` stays the GRID's third extent (`bh = H * B`) either
way — the two are independent.

`mheads` is the number of `bh`-slots that share one mask row-block, i.e. the TRUE head count — see
`flash_fwd!`'s addressing comment for why this cannot simply be `nh`. It defaults to `heads`, which
reproduces the previous behaviour exactly: one mask shared by everything. Pass the real head count
explicitly to get one mask per batch element instead (`mask` must then be `[n, n, heads ÷ mheads]`,
not `[n, n]`).
"""
function run_fwd!(o::MtlArray{T}, lse, q, k, v, bias, mask;
    tg::Int, kt::Int, n::Int, heads::Int, dim::Int, nouter::Int=n, nh::Int=1,
    mheads::Int=heads, scale::Float32=Float32(inv(sqrt(dim)))) where {T}
    _check_eltype(T)
    path, entry = build(T, dim, tg, kt)
    ctx, ngroups, ghost = LTA.kernel_context(MetalBackend(), n, heads, tg; nouter)
    tail = (o, lse, q, k, v, bias, mask, scale, NEG_INF, Int32(n), Int32(nouter), Int32(nh),
        Int32(mheads))
    # The context is uploaded verbatim as KA built it — its layout is KA's business, never ours.
    args = ghost ? (UInt32(0), tail...) : (UInt32(0), MtlArray([ctx]), tail...)
    launch(path, entry, args, ngroups, tg)
    return o
end

# Installed into `LTA._DEVICE_BINARY` by `__init__`, so the Reactant extension can name this binary
# in a `custom_call` without importing this module.
_binary(::Type{T}, dim::Int, tg::Int, kt::Int) where {T} = build(T, dim, tg, kt)

# Installed into `LTA._DEVICE_BINARY_BWD` by `__init__` — the backward analogue of `_binary`, for
# `tune_flash_kernel!` and for the traced backward the Reactant extension attaches as
# `enzyme.reverse` on its `stablehlo.custom_call`.
_binary_bwd(::Type{T}, dim::Int, tg::Int, kt::Int) where {T} = build_bwd(T, dim, tg, kt)

"""
    run_bwd!(dq, dk, dv, dbias, q, k, v, bias, mask, o, dO, lse;
        tg, kt, n, heads, dim, nouter=n, nh=1, mheads=heads, scale) -> (dq, dk, dv, dbias)

Run the backward on device arrays already in the kernel's `[dim, n, n, bh]` layout (`bias`/`mask` as
in [`run_fwd!`](@ref)). `o` and `lse` are `run_fwd!`'s own outputs for this `q`/`k`/`v`/`mask`/`bias`
— both are required residuals, not recomputed here (`o` builds `delta`; `lse` is used as-is).

Runs all three passes — [`LuxTriangleAttention.flash_bwd_dq!`](@ref), then
[`LuxTriangleAttention.flash_bwd_dk!`](@ref) and [`LuxTriangleAttention.flash_bwd_dv!`](@ref) —
since together they are the whole backward and nothing calls one without the others. `dbias` is
zeroed here before launch: `flash_bwd_dq!` accumulates into it with atomics, which requires starting
from zero (see its docstring).

`dK` and `dV` are two kernels rather than one for register-pressure reasons documented in
`src/kernels/flash/backward.jl`'s header; they are independent of each other and of `dq`, so the order
of the three launches is immaterial.

`delta` defaults to `flash_bwd_delta(o, dO)`, which assumes the packed `nh = 1` layout (channel axis
first, rank 4). Pass it explicitly when running at `nh > 1`, where it has to be laid out like `lse`
rather than like a reduction of `o`.
"""
function run_bwd!(dq::MtlArray{T}, dk::MtlArray{T}, dv::MtlArray{T}, dbias::MtlArray{Float32},
    q, k, v, bias, mask, o, dO, lse;
    tg::Int, kt::Int, n::Int, heads::Int, dim::Int, nouter::Int=n, nh::Int=1,
    mheads::Int=heads, scale::Float32=Float32(inv(sqrt(dim))),
    delta=LTA.flash_bwd_delta(o, dO)) where {T}
    _check_eltype(T)
    bins = build_bwd(T, dim, tg, kt)

    fill!(dbias, zero(Float32))
    ctx, ngroups, ghost = LTA.kernel_context(MetalBackend(), n, heads, tg; nouter)
    wrap(tail) = ghost ? (UInt32(0), tail...) : (UInt32(0), MtlArray([ctx]), tail...)
    trailing = (scale, Int32(n), Int32(nouter), Int32(nh), Int32(mheads))

    launch(bins.dq..., wrap((dq, dbias, q, k, v, bias, mask, dO, lse, delta, trailing...)),
        ngroups, tg)
    launch(bins.dk..., wrap((dk, q, k, v, bias, mask, dO, lse, delta, trailing...)),
        ngroups, tg)
    # `dV` takes neither `v` nor `delta` — see `flash_bwd_dv!`.
    launch(bins.dv..., wrap((dv, q, k, bias, mask, dO, lse, trailing...)), ngroups, tg)

    return dq, dk, dv, dbias
end
