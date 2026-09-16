module LuxTriangleAttentionEnzymeExt

using LuxTriangleAttention
using Enzyme
using Enzyme: EnzymeRules

const LTA = LuxTriangleAttention

# `q`, `k`, `v` and `bias` may each arrive active or constant depending on what the caller is
# differentiating with respect to; `mask` is boolean and never carries a gradient.
const _MaybeDup = Union{Enzyme.Const,Enzyme.Duplicated}

"""
    augmented_primal(config, ::Const{typeof(_sdpa_kernel)}, RT, backend, q, k, v, mask, bias)

Run the device forward and keep what the reverse needs.

The tape holds the primal operands plus `o` and `lse`. Both are genuine residuals rather than a
convenience: `o` is what `delta` is built from, and `lse` — the per-query log-sum-exp — is what lets
the backward recompute attention weights a tile at a time instead of the forward having to
materialise the full `Nk×Nq×H` weight tensor it deliberately never forms.

The shadow return is a tuple of zeroed buffers, one per primal result, because the primal returns
the `(output, lse)` pair. Nothing downstream differentiates `lse` — `_sdpa` takes `first` of the
pair — so its cotangent is always a materialised zero, but the calling convention still carries it.
"""
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Enzyme.Const{typeof(LTA._sdpa_kernel)},
    ::Type{RT},
    backend::Enzyme.Const{<:LTA.GPUBackend},
    q::_MaybeDup, k::_MaybeDup, v::_MaybeDup,
    mask::Enzyme.Annotation, bias::_MaybeDup) where {RT}

    o, lse = LTA._sdpa_kernel(backend.val, q.val, k.val, v.val, mask.val, bias.val)

    primal = EnzymeRules.needs_primal(config) ? (o, lse) : nothing
    shadow = EnzymeRules.needs_shadow(config) ? (zero(o), zero(lse)) : nothing
    tape = (q.val, k.val, v.val, mask.val, bias.val, o, lse, shadow)

    # Built through `augmented_rule_return_type` rather than by calling `AugmentedReturn` directly.
    # `primal` and `shadow` are conditional on the config, so their inferred type is a `Union` and
    # the resulting `AugmentedReturn` is a `UnionAll` — which Enzyme rejects outright, since it
    # needs the concrete return type to build the tape. The helper supplies exactly the type this
    # configuration expects.
    return EnzymeRules.augmented_rule_return_type(config, RT)(primal, shadow, tape)
end

"""
    reverse(config, ::Const{typeof(_sdpa_kernel)}, dret, tape, backend, q, k, v, mask, bias)

Run the entry's backward stages and accumulate into the argument shadows.

`_sdpa_kernel_bwd!` returns fresh gradient buffers rather than accumulating in place, so they are
added into `.dval` here — Enzyme's contract is accumulation, not assignment, since the same argument
can be reached by more than one path through the differentiated function.

Returns one element per ARGUMENT (`backend`, `q`, `k`, `v`, `mask`, `bias`), all `nothing`: every
active argument is `Duplicated`, so its gradient goes into its shadow rather than being returned.
Getting this arity wrong is a loud `ReverseRuleReturnError` rather than a silent miscompile, which
is the one part of writing a rule by hand that checks itself.
"""
function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Enzyme.Const{typeof(LTA._sdpa_kernel)},
    dret, tape,
    backend::Enzyme.Const{<:LTA.GPUBackend},
    q::_MaybeDup, k::_MaybeDup, v::_MaybeDup,
    mask::Enzyme.Annotation, bias::_MaybeDup)

    qv, kv, vv, mv, bv, o, lse, shadow = tape

    # The cotangent of `o`. `dret` carries it when Enzyme threads one through; otherwise the shadow
    # this rule allocated in `augmented_primal` IS the accumulator the caller wrote into.
    dO = _dout(dret, shadow)

    dq, dk, dv, dbias = LTA._sdpa_kernel_bwd!(qv, kv, vv, mv, bv, o, lse, dO)

    q isa Enzyme.Duplicated && (q.dval .+= dq)
    k isa Enzyme.Duplicated && (k.dval .+= dk)
    v isa Enzyme.Duplicated && (v.dval .+= dv)
    bias isa Enzyme.Duplicated && (bias.dval .+= reshape(dbias, size(bias.val)))

    return (nothing, nothing, nothing, nothing, nothing, nothing)
end

# The `(output, lse)` pair means the incoming cotangent is a 2-tuple; only the first half is ever
# meaningful (`lse` is a residual nothing downstream reads). Falls back to the rule's own shadow when
# Enzyme passes no explicit cotangent.
_dout(dret::Tuple, shadow) = first(dret)
_dout(dret, shadow) = shadow === nothing ? error(
    "no cotangent available for `_sdpa_kernel`; this is an Enzyme configuration this rule does " *
    "not handle") : first(shadow)

end # module
