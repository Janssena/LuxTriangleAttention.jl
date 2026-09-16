# Running a registry entry on Metal, without knowing which kernel it is.
#
# Everything that varies between kernels — how many launches the backward takes, which gradient each
# writes, what residuals it reads, the grid — is DATA on the entry (`KernelStage`). This file is the
# interpreter for that data: it materialises each role, binds them in the order `abi_order` derives,
# and launches. It contains no knowledge of our own flash kernel beyond the two preamble slots that
# kernel happens to need.
#
# The one thing NOT derived from the entry is the buffer ORDER, which comes from the fixed rank
# table in `src/kernels/registry.jl`. That is deliberate: a wrong slot table binds without complaint
# and produces wrong numbers rather than an error, so it is not something an entry gets to specify.

"""
    _ProblemCtx

Everything a role can be resolved against for one launch: the operands, the residuals, the shapes,
and the allocated outputs.

Built once per launch and passed to [`_role_value`](@ref) for every slot, so a role's meaning is
defined in exactly one place rather than once per stage.
"""
struct _ProblemCtx{T}
    q::Any
    k::Any
    v::Any
    bias::Any
    mask::Any        # already negated to the kernel's DROP convention
    o::Any
    lse::Any
    delta::Any
    dO::Any
    outs::Dict{Symbol,Any}
    n::Int
    nouter::Int      # the outer extent (`Nj`); equals `n` for triangle attention
    dim::Int
    H::Int
    B::Int
    bh::Int
    tg::Int
    scale::Float32
end

const NEG_INF_F32 = -floatmax(Float32)

"""
    _role_value(role, ctx) -> buffer or scalar

Resolve one role against a launch context. The single definition of what each role MEANS.

Output roles come from `ctx.outs`, which the launcher allocated before binding; input roles come
from the operands and residuals. `:o_in`/`:lse_in`/`:delta_in` are the reading names for values some
stage produced (or the runtime supplied) — see `_INPUT_SOURCE` in the registry for why the reading
and writing names differ.
"""
function _role_value(role::Symbol, ctx::_ProblemCtx)
    haskey(ctx.outs, role) && return ctx.outs[role]
    role === :q && return ctx.q
    role === :k && return ctx.k
    role === :v && return ctx.v
    role === :bias && return ctx.bias
    role === :mask && return ctx.mask
    role === :o_in && return ctx.o
    role === :lse_in && return ctx.lse
    role === :delta_in && return ctx.delta
    role === :dO && return ctx.dO
    role === :scale && return ctx.scale
    role === :neg_inf && return NEG_INF_F32
    role === :n && return Int32(ctx.n)
    role === :nouter && return Int32(ctx.nouter)
    role === :nh && return Int32(ctx.H)
    role === :mheads && return Int32(ctx.H)
    role === :h && return Int32(ctx.H)
    role === :dim && return Int32(ctx.dim)
    throw(ArgumentError("role `:$role` has no value in this launch context"))
end

"""
    _allocate_output(role, ctx) -> MtlArray

Allocate a buffer for an output role, in the shape and precision that role requires.

`dbias` is ZEROED and the rest are not: `flash_bwd_dq!` accumulates into `dbias` with atomics, which
is only correct starting from zero (see its docstring). Every other output is fully written by its
kernel, so zeroing them would be pure cost.

`lse` and `delta` are `Float32` regardless of the storage type — they are the residuals the backward
reconstructs probabilities from, and narrowing them would defeat the numerical-stability mechanism
the online softmax is built on. See note 3 at the top of `flash/forward.jl`.
"""
function _allocate_output(role::Symbol, ctx::_ProblemCtx{T}) where {T}
    n, nouter, H, B = ctx.n, ctx.nouter, ctx.H, ctx.B
    role === :o && return similar(ctx.q)
    role === :lse && return similar(ctx.q, Float32, (1, n, H, nouter, B))
    role === :delta && return similar(ctx.q, Float32, (1, n, H, nouter, B))
    role === :dq && return similar(ctx.q)
    role === :dk && return similar(ctx.k)
    role === :dv && return similar(ctx.v)
    role === :dbias && return Metal.zeros(Float32, size(ctx.bias))
    throw(ArgumentError("no allocation rule for output role `:$role`"))
end

"""
    _resolve_grid(axis, ctx, ngroups) -> Int

One grid extent. `:ka_ngroups` is KernelAbstractions' own 1-D group count, already computed for this
problem; the rest are plain functions of the problem size.
"""
function _resolve_grid(axis::Symbol, ctx::_ProblemCtx, ngroups::Int)
    axis === :ka_ngroups && return ngroups
    axis === :one && return 1
    axis === :n && return ctx.n
    axis === :bh && return ctx.bh
    axis === :cdiv_n_bj && return cld(ctx.n, ctx.tg)
    axis === :cdiv_n_bk && return cld(ctx.n, ctx.tg)
    throw(ArgumentError("unknown grid axis `:$axis`"))
end

"""
    _run_stage!(entry, stage, ctx) -> ctx

Bind and launch one stage, allocating its outputs into `ctx.outs` first.

Outputs are allocated here rather than up front so that a role written by a later stage does not
exist while an earlier one runs — which is what makes the registry's "read only what an earlier
stage wrote" check meaningful at runtime rather than merely on paper.
"""
function _run_stage!(entry, stage, ctx::_ProblemCtx{T}) where {T}
    for r in stage.outputs
        haskey(ctx.outs, r) || (ctx.outs[r] = _allocate_output(r, ctx))
    end

    kactx, ngroups, ghost = LTA.kernel_context(MetalBackend(), ctx.n, ctx.bh, stage.threadgroup;
        nouter=ctx.nouter)

    args = Any[]
    for (kind, role) in LTA.abi_order(entry, stage)
        if kind === :bytes
            # The preamble. `:kernel_state` is GPUCompiler's slot; `:ka_context` is KA's launch
            # context, uploaded verbatim as KA built it — its layout is KA's business, never ours —
            # and skipped entirely when it is a ghost type occupying no buffer slot.
            role === :kernel_state && (push!(args, UInt32(0)); continue)
            role === :ka_context && (ghost || push!(args, kactx); continue)
            throw(ArgumentError("unknown preamble slot `:$role`"))
        end
        push!(args, _role_value(role, ctx))
    end

    g = ntuple(i -> _resolve_grid(stage.grid[i], ctx, ngroups), 3)
    launch(LTA.stage_path(entry, stage), stage.entry, args, prod(g), stage.threadgroup)
    return ctx
end

"""
    _make_ctx(entry, stages, q, k, v, mask, bias; o, lse, dO) -> _ProblemCtx

Build the launch context from LuxTriangleAttention's own `[hd, H, Nq, Nj, B]` operands.

# Layout
No permutation happens for an entry whose `layout` already is this package's own — which is the case
for `:$(LTA.RESERVED_KERNEL_NAME)`. An `MtlArray` of that size is column-major, so `hd` is already
the fastest axis with the head axis next, and the kernel addresses it by being told a head stride of
`nh = H` (see `flash_fwd!`). An entry declaring a different layout is rejected here rather than
silently mis-addressed: permuting into an arbitrary declared layout is real data movement and is not
yet implemented, and producing wrong numbers instead of saying so would be much worse.

`mask` is LuxTriangleAttention's KEEP mask and is negated here — the kernel's convention is that
`true` means *drop*. It is a KEY mask per outer index, `[Nq, Nj, B]` once reshaped, and varies per
batch element but never per head, which is what `mheads` is for.

`Nq ≠ Nj` is any attention that is not triangle attention: `nouter = Nj` is passed as its own
argument. `bias === nothing` (e.g. MSA column attention) becomes a zero bias, which adds nothing
to any score; it costs `Nq²·H·B` elements, against the `Nq²·H·Nj·B` score tensor it replaces.
"""
function _make_ctx(entry, q, k, v, mask, bias; o=nothing, lse=nothing, dO=nothing, delta=nothing)
    entry.layout == LTA._DEFAULT_LAYOUT || throw(ArgumentError(
        "kernel `:$(entry.name)` declares layout $(entry.layout); this launcher currently only " *
        "supports this package's own $(LTA._DEFAULT_LAYOUT). Registering a kernel in another " *
        "layout needs the permuting path, which is not implemented."))

    dim, H, Nq, Nj, B = size(q)
    T = eltype(q)
    _check_eltype(T)

    n, nouter, bh = Nq, Nj, H * B
    md = mask === nothing ? MtlArray(falses(n, nouter, B)) : (.!reshape(mask, n, nouter, B))
    bias = bias === nothing ? Metal.zeros(T, n, n, H, B) : bias
    tg = 0  # filled per stage; only `:cdiv_*` grids read it, which our kernel does not use

    return _ProblemCtx{T}(q, k, v, bias, md, o, lse, delta, dO,
        Dict{Symbol,Any}(), n, nouter, dim, H, B, bh, tg, Float32(inv(sqrt(dim))))
end

"""
    LuxTriangleAttention._sdpa_kernel_launch(backend, entry, stages, q, k, v, mask, bias)

The concrete-array forward: run the entry's `:fwd` stage and return `(output, lse)`.

Dispatches on `MtlArray` as well as the backend tag so the Reactant extension can add the traced
counterpart for the same tag without either extension knowing about the other.
"""
function LTA._sdpa_kernel_launch(::LTA.MetalBackendTag, entry, stages,
    q::MtlArray{<:Any,5}, k::MtlArray, v::MtlArray, mask, bias)
    fwd = LTA.find_stage(stages, :fwd)
    fwd === nothing && throw(ArgumentError("kernel `:$(entry.name)` has no `:fwd` stage"))
    LTA.check_nouter(entry, (fwd,), size(q, 3), size(q, 4))
    ctx = _make_ctx(entry, q, k, v, mask, bias)
    _run_stage!(entry, fwd, ctx)
    return ctx.outs[:o], ctx.outs[:lse]
end

"""
    LuxTriangleAttention._sdpa_kernel_bwd!(q, k, v, mask, bias, o, lse, dO) -> (dq, dk, dv, dbias)

The concrete-array backward: run every non-`:fwd` stage of the selected entry, in order.

The order is the entry's, not ours — it is what makes both our `dq`(+`dbias`) / `dk` / `dv` split
and a kernel that splits differently (trifast's `dq`(+`delta`) / `dk`+`dv` / `dbias`) run through
this same loop. `delta` is supplied up front here because our own stages read it without any stage
producing it; an entry whose first stage WRITES `delta` simply overwrites this one.
"""
function LTA._sdpa_kernel_bwd!(q::MtlArray{<:Any,5}, k, v, mask, bias, o, lse, dO)
    be = LTA.backend_of(q)
    entry = LTA.kernel_entry()
    T = eltype(q)
    dim, H, Nq, Nj, B = size(q)
    stages = LTA.resolve_variant(entry, T, dim, Nq)
    stages === nothing && (stages = LTA._ensure_variant(entry, T, dim, Nq))
    LTA.check_nouter(entry, stages, Nq, Nj)

    # `flash_bwd_delta` assumes the packed layout (channel axis first, rank 4). Here the channel
    # axis is dim 1 of a rank-5 array and the result has to be laid out like `lse`, so the reduction
    # is written out rather than reused.
    delta = permutedims(sum(o .* dO; dims=1), (1, 3, 2, 4, 5))

    ctx = _make_ctx(entry, q, k, v, mask, bias; o, lse, dO, delta)
    for s in stages
        s.name === :fwd && continue
        _run_stage!(entry, s, ctx)
    end
    return (ctx.outs[:dq], ctx.outs[:dk], ctx.outs[:dv], ctx.outs[:dbias])
end
