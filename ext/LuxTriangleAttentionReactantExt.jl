module LuxTriangleAttentionReactantExt

using LuxTriangleAttention: LuxTriangleAttention, Attention
using Reactant: Reactant, AnyTracedRArray, TracedRNumber
using Reactant.ReactantCore: @trace

LuxTriangleAttention._realtype(::Type{TracedRNumber{T}}) where T = T

"""
    LuxTriangleAttention.backend_of(::AnyTracedRArray)

Which device a traced array will execute on.

This method is the reason `backend_of` exists as a trait at all: a `TracedRArray` is the same Julia
type whether it will run on CPU or GPU, so the array cannot answer the question — only the
compiling client can. Everything else in the routing dispatches on the array type quite happily.

An unrecognised platform maps to `UnknownBackend`, which falls back rather than failing.
"""
LuxTriangleAttention.backend_of(::Union{Reactant.AnyConcreteRArray, Reactant.AnyTracedRArray}) = 
    _reactant_backend()

function _reactant_backend()
    platform = try
        Reactant.XLA.platform_name(Reactant.XLA.default_backend())
    catch
        return LuxTriangleAttention.UnknownBackend()
    end
    platform == "mps" && return LuxTriangleAttention.MetalBackendTag()
    platform == "cuda" && return LuxTriangleAttention.CUDABackendTag()
    platform == "rocm" && return LuxTriangleAttention.ROCmBackendTag()
    platform == "cpu" && return LuxTriangleAttention.CPUBackend()
    return LuxTriangleAttention.UnknownBackend()
end

# Reactant traces every element, so a boolean mask under `Reactant.@compile`
# arrives as `TracedRArray{Bool,N} <: AbstractArray{TracedRNumber{Bool},N}`,
# not `AbstractArray{Bool,N}` — plain `mask::AbstractArray{Bool}` dispatch in
# the base package never matches it. `Reactant.AnyTracedRArray{T,N}` is the
# alias Reactant provides for exactly this ("anything holding traced `T`s",
# covering `TracedRArray` itself and reshapes/views of it); give traced Bool
# masks the same codepaths as plain `Bool` masks.
LuxTriangleAttention.ending_permute(mask::AnyTracedRArray{Bool,3}) =
    permutedims(mask, (2, 1, 3))

(l::Attention)(x::Union{<:Tuple,AnyTracedRArray{T}}, mask::AnyTracedRArray{Bool}, ps, st) where T =
    l(x, nothing, mask, ps, st)




# The rolled `@trace` chunking path (`_sdpa_stable`, `chunked_compile_options`, `check_rolled`,
# `_chunked_sdpa` and its helpers) lives in `ext/reactant/chunk.jl`.
include("reactant/chunk.jl")


# `_mat` is a general Reactant helper used below by the KA-traced device kernel path.
_mat(x) = Reactant.TracedUtils.materialize_traced_array(x)

# ---------------------------------------------------------------------------------------------
# `_sdpa_kernel(q, k, v, mask, bias)` — the base package's generic entry point in
# `src/kernels/backend.jl` — needs no override here. It already dispatches on `backend_of(q)`,
# which this extension gives a real answer for traced arrays (see `_reactant_backend` below), and
# the GPU branch reaches `_sdpa_kernel_launch`, which THIS file's method for `AnyTracedRArray`
# picks up without either side needing to know about the other.

# Everything specific to writing a `stablehlo.custom_call` (`_embed_custom_call`,
# `_metal_kernel_lib`, `_attach_reverse_rule!`, `_le_bytes`, `_fresh`) lives in
# `ext/reactant/custom_call.jl`.
include("reactant/custom_call.jl")

"""
    _rev(x)

Reverse a traced array's dimensions — the ONLY layout conversion this path performs, and a free one.

Reactant lays a traced value out row-major over its Julia size, so the last Julia dimension is the
fastest-varying one in memory. LuxTriangleAttention's `[hd, H, Nq, Nj, B]` therefore already has
`hd` fastest, `H` next, then `Nq`, `Nj`, `B` — which is exactly the order the kernel addresses (see
`flash_fwd!`'s `nh` stride). The two disagree only about which END of the dimension list is written
first, so reversing is a pure relabelling: XLA folds it into the parameter or result layout rather
than copying, and it survives optimisation as nothing at all.

This replaced a genuine permutation, `(5, 2, 4, 3, 1)`, that moved the head axis out to join the
batch and produced the packed `[dim, n, n, bh]` buffers the direct-Metal path uses. That one was
real data movement: three full-size input copies plus one to bring `o` back, which is what made the
XLA path cost twice the device memory of the direct path at the same `N`. Teaching the kernel a head
stride was the cheaper half of that trade.
"""
_rev(x) = _mat(permutedims(x, ntuple(i -> ndims(x) - i + 1, ndims(x))))


# =================================================================================================
# Running a registry entry from a traced graph
# =================================================================================================
#
# The concrete-array route (`MtlArray` in, `MtlArray` out) is the backend extension's; this is the
# traced one, where the same cached binaries are NAMED by `stablehlo.custom_call`s instead of being
# launched directly. Both walk the same `KernelEntry` stages and bind the same derived ABI order, so
# a kernel registered once works on both paths without knowing either exists.
#
# !!! warning "The reverse path here is not yet exercised"
#     `_attach_reverse_rule!` writes an `enzyme.reverse` attribute onto the custom call. Enzyme-XLA
#     does not yet honour that attribute for custom calls, so the traced BACKWARD below is written
#     against the documented contract but cannot be run today. The traced FORWARD is exercised. The
#     concrete-array backward — same stages, same binaries, via `LuxTriangleAttentionEnzymeExt` — IS
#     verified against a host reference, so what is untested here is the plumbing, not the kernels.

"""
    _kernel_shape(role, dims) -> Tuple

The XLA result shape for an output role, in the REVERSED layout traced values use.

Reactant lays a traced value out row-major over its Julia size, so LuxTriangleAttention's
`[hd, H, Nq, Nj, B]` already has `hd` fastest — the order the kernel addresses. The two disagree only
about which end of the dimension list is written first, so every shape here is its Julia counterpart
reversed, and `_rev` converts between them as a pure relabelling XLA folds away. See [`_rev`](@ref).
"""
function _kernel_shape(role::Symbol, d)
    hd, H, Nq, Nj, B = d.hd, d.H, d.Nq, d.Nj, d.B
    role in (:o, :dq, :dk, :dv) && return (B, Nj, Nq, H, hd)
    role in (:lse, :delta) && return (B, Nj, H, Nq, 1)
    # The bias is `[Nq, Nq, H, B]` — keys × queries, with no outer axis — so its gradient is too.
    role === :dbias && return (B, 1, H, Nq, Nq)
    throw(ArgumentError("no traced result shape for output role `:$role`"))
end

"""
    _scalar_bytes(role, d) -> String

The little-endian byte payload for a scalar role, for the `:bytes` slot it occupies in the buffer
table. The types must match the compiled kernel's signature exactly — `Float32` for `scale` and
`neg_inf`, `Int32` for the counts — since a `:bytes` slot is copied verbatim and a wrong width
shifts everything after it.
"""
function _scalar_bytes(role::Symbol, d)
    role === :scale && return _le_bytes(Float32(inv(sqrt(d.hd))))
    role === :neg_inf && return _le_bytes(-floatmax(Float32))
    role === :n && return _le_bytes(Int32(d.Nq))
    role === :nouter && return _le_bytes(Int32(d.Nj))
    role in (:nh, :mheads, :h) && return _le_bytes(Int32(d.H))
    role === :dim && return _le_bytes(Int32(d.hd))
    throw(ArgumentError("no byte encoding for scalar role `:$role`"))
end

"""
    _emit_stage(entry, stage, vals, d; reverse=nothing, active_operands=nothing) -> results

Emit one stage as a `stablehlo.custom_call`, building its buffer table from [`abi_order`](@ref).

This is the traced counterpart of `_run_stage!` in `ext/metal/launch.jl`, and it derives the slot
table from the same source. Writing the table out by hand at each call site — which is what this
path used to do — was a chance to mis-order a slot per kernel, silently, since a wrong table binds
without complaint and produces wrong numbers rather than an error.
"""
function _emit_stage(entry, stage, vals::Dict{Symbol,Any}, d;
        reverse=nothing, active_operands=nothing)
    slots = LuxTriangleAttention.abi_order(entry, stage)

    kactx, ngroups, ghost = LuxTriangleAttention.kernel_context(
        _ka_backend_object(LuxTriangleAttention.MetalBackendTag()), d.Nq, d.H * d.B,
        stage.threadgroup; nouter=d.Nj)

    operands = Any[]
    outroles = Symbol[]
    layout = Any[]
    for (kind, role) in slots
        if kind === :bytes
            role === :kernel_state && (push!(layout, (:bytes, _le_bytes(UInt32(0)))); continue)
            role === :ka_context && (ghost || push!(layout, (:bytes, _le_bytes(kactx))); continue)
            throw(ArgumentError("unknown preamble slot `:$role`"))
        elseif kind === :out
            # `dbias` is zero-initialised by the runtime, as the concrete launcher's
            # `_allocate_output` does: `flash_bwd_dq!` ACCUMULATES it with atomics, and jax-mps
            # otherwise hands custom-call results over uninitialised. See docs/BUG_traced_dbias.md.
            push!(layout, (role === :dbias ? :out_zero : :out, length(outroles)))
            push!(outroles, role)
        elseif kind === :in
            push!(layout, (:in, length(operands)))
            push!(operands, vals[role])
        else
            push!(layout, (:bytes, _scalar_bytes(role, d)))
        end
    end

    shapes = [_kernel_shape(r, d) for r in outroles]
    grid = ntuple(i -> _grid_extent(stage.grid[i], d, ngroups, stage.threadgroup), 3)
    results = _metal_kernel_lib(stage.entry, LuxTriangleAttention.stage_path(entry, stage),
        Tuple(operands), shapes, layout;
        grid=(prod(grid), 1, 1), threadgroup=stage.threadgroup,
        reverse, active_operands)
    return outroles, results
end

_grid_extent(axis::Symbol, d, ngroups, tg) =
    axis === :ka_ngroups ? ngroups :
    axis === :one ? 1 :
    axis === :n ? d.Nq :
    axis === :bh ? d.H * d.B :
    axis in (:cdiv_n_bj, :cdiv_n_bk) ? cld(d.Nq, tg) :
    throw(ArgumentError("unknown grid axis `:$axis`"))

# The bias and mask operands in the kernel's (reversed) layout. The bias is `[Nq, Nq, H, B]` —
# keys × queries, shared by every outer index — and `nothing` becomes zeros, which add nothing to
# any score (MSA column attention has no bias). The mask is negated to the kernel's DROP convention
# and is a key mask per outer index, `[Nq, Nj, B]`; `nothing` drops nothing.
_kernel_bias(::Nothing, q, d) =
    _mat(Reactant.Ops.fill(zero(Reactant.unwrapped_eltype(q)), (d.B, d.H, d.Nq, d.Nq)))
_kernel_bias(bias, q, d) = _rev(_mat(reshape(bias, size(bias, 1), size(bias, 2), d.H, d.B)))
_kernel_mask(::Nothing, d) = _mat(Reactant.Ops.fill(false, (d.B, d.Nj, d.Nq)))
_kernel_mask(mask, d) = _rev(_mat(reshape(.!mask, d.Nq, d.Nj, d.B)))

"""
    LuxTriangleAttention._sdpa_kernel_launch(backend, entry, stages, q, k, v, mask, bias)

The traced forward: emit the entry's `:fwd` stage as a custom call against its cached binary.

Every operand reaches the kernel through [`_rev`](@ref) and nothing else, so this path allocates
exactly what the direct-Metal path does: the two results, and no copies of the inputs.

`active_operands = (0,1,2,3)` names `q,k,v,bias` as differentiable; `mask` (slot 4) is left out,
matching `_attach_reverse_rule!`'s convention for a non-differentiable operand — a boolean mask
carries no gradient.
"""
function LuxTriangleAttention._sdpa_kernel_launch(
        ::LuxTriangleAttention.MetalBackendTag, entry, stages,
        q::Reactant.AnyTracedRArray{<:Any,5}, k, v, mask, bias)
    fwd = LuxTriangleAttention.find_stage(stages, :fwd)
    fwd === nothing && throw(ArgumentError("kernel `:$(entry.name)` has no `:fwd` stage"))

    hd, H, Nq, Nj, B = size(q)
    LuxTriangleAttention.check_nouter(entry, (fwd,), Nq, Nj)
    d = (; hd, H, Nq, Nj, B)

    vals = Dict{Symbol,Any}(
        :q => _rev(q), :k => _rev(k), :v => _rev(v),
        :bias => _kernel_bias(bias, q, d), :mask => _kernel_mask(mask, d))

    outroles, results = _emit_stage(entry, fwd, vals, d;
        reverse=(args...) -> _traced_reverse(entry, stages, d, args...),
        active_operands=(0, 1, 2, 3))
    out = Dict(zip(outroles, results))
    return _rev(out[:o]), _rev(out[:lse])
end

"""
    _traced_reverse(entry, stages, d, qt, kt_, vt, bt, mt, o, lse, dO, dlse) -> (dq, dk, dv, dbias)

The `vjp` attached to the forward custom call via `_attach_reverse_rule!`.

Runs every non-`:fwd` stage of the entry as its own custom call, in the order the entry declares —
the same walk `_sdpa_kernel_bwd!` performs on concrete arrays, against the same binaries.

Every argument is already in the KERNEL's layout: `qt…mt` are the forward's own converted operands
and `o`/`lse` its RAW results (`_attach_reverse_rule!` receives them before `_rev` runs), so nothing
is converted again here. `dlse`'s VALUE is never used — nothing downstream differentiates `lse`, so
Enzyme always materialises a zero cotangent for it — but the calling convention still passes it
positionally; see `_attach_reverse_rule!`'s docstring.

Returns four cotangents, matching `active_operands = (0,1,2,3)` at the call site.
"""
function _traced_reverse(entry, stages, d, qt, kt_, vt, bt, mt, o, lse, dO, dlse)
    # `delta` is a reduction over the channel axis laid out like `lse`. In the reversed traced
    # layout the channel axis is dim 5, and reducing it leaves the head axis where `o` has it rather
    # than where `lse` does, so the swap back is part of the reduction.
    delta = _mat(permutedims(dropdims(sum(o .* dO; dims=5); dims=5), (1, 2, 4, 3)))

    vals = Dict{Symbol,Any}(
        :q => qt, :k => kt_, :v => vt, :bias => bt, :mask => mt,
        :dO => dO, :o_in => o, :lse_in => lse, :delta_in => delta)

    grads = Dict{Symbol,Any}()
    for s in stages
        s.name === :fwd && continue
        outroles, results = _emit_stage(entry, s, vals, d)
        for (r, res) in zip(outroles, results)
            grads[r] = res
            # A stage that PRODUCES a residual makes it readable by later stages, exactly as on the
            # concrete path — this is what lets a kernel whose first backward stage computes `delta`
            # (rather than taking it precomputed) work here unchanged.
            haskey(LuxTriangleAttention._INPUT_SOURCE, r) && (vals[r] = res)
            r === :delta && (vals[:delta_in] = res)
        end
    end
    return grads[:dq], grads[:dk], grads[:dv], grads[:dbias]
end

"""
    LuxTriangleAttention._sdpa_kernel_bwd!(q, k, v, mask, bias, o, lse, dO)

The traced backward, reached when something differentiates the device path outside the custom call's
own `enzyme.reverse`. Shares [`_traced_reverse`](@ref)'s walk after converting operands into the
kernel's layout.

Operands go in through [`_rev`](@ref) and the gradients come back out through it, so this returns
LuxTriangleAttention-layout arrays — the same contract the concrete-array method in the Metal
extension honours. `_traced_reverse` itself speaks the kernel's layout on both sides, because the
`enzyme.reverse` function it also serves is handed operands that are already converted.
"""
function LuxTriangleAttention._sdpa_kernel_bwd!(q::Reactant.AnyTracedRArray{<:Any,5},
        k, v, mask, bias, o, lse, dO)
    entry = LuxTriangleAttention.kernel_entry()
    hd, H, Nq, Nj, B = size(q)
    d = (; hd, H, Nq, Nj, B)
    T = Reactant.unwrapped_eltype(q)
    stages = LuxTriangleAttention.resolve_variant(entry, T, hd, Nq)
    stages === nothing && (stages = LuxTriangleAttention._ensure_variant(entry, T, hd, Nq))
    LuxTriangleAttention.check_nouter(entry, stages, Nq, Nj)
    dq, dk, dv, dbias = _traced_reverse(entry, stages, d,
        _rev(q), _rev(k), _rev(v), _kernel_bias(bias, q, d), _kernel_mask(mask, d),
        _rev(o), _rev(lse), _rev(dO), nothing)
    return _rev(dq), _rev(dk), _rev(dv), _rev(dbias)
end

# KA needs a backend OBJECT (not our tag) to build a launch context. Only the loaded backend
# extension can supply one, so it registers a constructor here.
_ka_backend_object(::LuxTriangleAttention.KernelBackend) =
    LuxTriangleAttention._KA_BACKEND_OBJECT[]()

"""
    __init__()

Tell the base package how to detect that it is tracing.

`_ensure_variant` must refuse to lazily tune the built-in kernel mid-trace: tuning builds real
device arrays and times real launches, which `call_with_reactant`'s overlay would capture into the
graph, at which point the device array constructor rejects the traced element type. Nothing is
registered here — the registry is populated by `tune_flash_kernel!` and persists on disk.
"""
function __init__()
    LuxTriangleAttention._TRACING[] = () -> Reactant.within_compile()
    return nothing
end

end # module LuxTriangleAttentionReactantExt
