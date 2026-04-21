struct TriangleMultiplication{LN, CORE} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :core)}
    layer_norm::LN
    core::CORE
end

"""
    TriangleMultiplication(chn_in::Int, hidden_chn::Int; is_outgoing=true, layernorm_eps=1f-5, kwargs...)

Takes x ~ [C, N, N, B] and returns y ~ [C, N, N, B].
If `is_outgoing=true`, performs Outgoing multiplication (Algorithm 11).
If `is_outgoing=false`, performs Incoming multiplication (Algorithm 12).
"""
function TriangleMultiplication(
    chn_in::Int, chn_hidden::Int;
    is_outgoing::Union{Bool, StaticBool}=static(true),
    layernorm_eps=1f-5, kwargs...
)
    return TriangleMultiplication(
        Lux.LayerNorm((chn_in, 1, 1); dims=1, epsilon=layernorm_eps),
        TriMulCore(chn_in, chn_hidden; is_outgoing, layernorm_eps, kwargs...)
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

__trimul_permute_maybe(::False, x) = x
__trimul_permute_maybe(::False, args::Vararg) = args

__trimul_permute_maybe(::True, x) = __ingoing_permute(x)
__trimul_permute_maybe(::True, args::Vararg) = map(__ingoing_permute, args)

__ingoing_permute(::Nothing) = nothing
__ingoing_permute(x::AbstractArray{T,3}) where T = permutedims(x, (2, 1, 3))
__ingoing_permute(x::AbstractArray{T,4}) where T = permutedims(x, (1, 3, 2, 4))