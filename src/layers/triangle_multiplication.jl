"""
    TriangleMultiplication(chn_in, hidden_chn; is_outgoing=true, kwargs...)

Implementation of Triangle Multiplication (Algorithm 11 & 12 from AlphaFold2).
Updates the pair representation by combining edges sharing a common node.

## Arguments
- `chn_in`: Input channels.
- `chn_hidden`: Hidden channels for internal projections.

## Keyword Arguments
- `is_outgoing`: If `true`, performs "Outgoing" multiplication (Algorithm 11). 
  If `false`, performs "Incoming" multiplication (Algorithm 12).
- `layernorm_eps`: Epsilon for LayerNorm.
- `rank`: The rank (ndims) of the input.
- `kwargs...`: Passed to `TriMulCore`.

## Inputs
- `x`: Input tensor with shape `[C, N, N, B]`.
- `mask`: Optional mask with shape `[N, N, B]`.

## Returns
- `y`: Output tensor with shape `[C, N, N, B]`.
- `st`: Updated state.
"""
struct TriangleMultiplication{LN, CORE} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :core)}
    layer_norm::LN
    core::CORE
end


function TriangleMultiplication(
    chn_in::Int, chn_hidden::Int;
    is_outgoing::Union{Bool, StaticBool}=static(true), rank=4,
    layernorm_eps=1f-5, kwargs...
)
    return TriangleMultiplication(
        Lux.LayerNorm((chn_in, ntuple(one, rank-2)...); dims=1, epsilon=layernorm_eps),
        TriMulCore(chn_in, chn_hidden; is_outgoing, layernorm_eps, rank, kwargs...)
    )
end

(m::TriangleMultiplication)(x::AbstractArray, ps, st) = m(x, nothing, ps, st)
(m::TriangleMultiplication)(inputs::NamedTuple, ps, st) = m(
    inputs.x, 
    get(inputs, :mask, nothing), 
    ps, st
)

function (m::TriangleMultiplication)(x::AbstractArray, mask, ps, st)
    x̃, layer_norm = m.layer_norm(x, ps.layer_norm, st.layer_norm)
    out, core = m.core(x̃, mask, ps.core, st.core)
    return out, merge(st, (; layer_norm, core))
end