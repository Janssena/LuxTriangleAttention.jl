struct TriangleAttention{P<:StaticBool,LN,B,MHA} <: Lux.AbstractLuxContainerLayer{(:layer_norm,:linear,:mha)}
    permute::P
    layer_norm::LN
    linear::B
    mha::MHA
end

"""
    TriangleAttention(chn_in::Int, num_heads::Int; starting=true, inf=_safe_inf(Float32))

Takes x ~ [C, N, N, B] and optionally a mask ~ [N, N, B] and returns y ~ [C, N, N, B]
"""
function TriangleAttention(
    chn_in::Int, chn_hidden::Int, num_heads::Int;
    is_starting::Union{Bool, StaticBool}=static(true), use_bias=true, layernorm_eps=1f-5, kwargs...
) 
    return TriangleAttention(
        static(!is_starting),
        Lux.LayerNorm((chn_in, 1, 1); dims=1, epsilon=layernorm_eps), # AF2, AF3, and Boltz2 all use use_bias=true
        Lux.Dense(chn_in, num_heads; use_bias), # AF2 uses use_bias=false
        TriAttnCore(chn_in, chn_hidden, num_heads; kwargs...)
    )
end

(m::TriangleAttention)(inputs::NamedTuple, ps, st) = _triangle_attention_forward(
    m,
    inputs.x, 
    get(inputs, :mask, nothing), 
    ps, st
)
(m::TriangleAttention)(x::AbstractArray, ps, st) = m(x, nothing, ps, st)

(m::TriangleAttention)(x::AbstractArray, mask, ps, st) = 
    _triangle_attention_forward(m, x, mask, ps, st)

function _triangle_attention_forward(m::TriangleAttention, x, mask, ps, st)
    x, mask = __tri_attn_permute_maybe(m.permute, x, mask)
    x̃, layer_norm = m.layer_norm(x, ps.layer_norm, st.layer_norm) # [C, N, N, B]
    
    bias, linear = m.linear(x̃, ps.linear, st.linear) # [H, N, N, B]

    y, mha = m.mha(x̃, bias, mask, ps.mha, st.mha) # [C, N, N, B]
    
    y = __tri_attn_permute_maybe(m.permute, y)

    return y, merge(st, (; layer_norm, linear, mha))
end

# starting = true -> permute = false -> no permutation
__tri_attn_permute_maybe(::False, x) = x
__tri_attn_permute_maybe(::False, x, mask) = x, mask

# starting = false -> permute = true ->  swap the i and j dimensions
__tri_attn_permute_maybe(::True, x) = __ending_permute(x)
__tri_attn_permute_maybe(::True, args::Vararg) = map(__ending_permute, args)

__ending_permute(::Nothing) = nothing
__ending_permute(x::AbstractArray{T,4}) where T = permutedims(x, (1, 3, 2, 4)) # swap i and j in [C, N, N, B]
__ending_permute(mask::AbstractArray{T,3}) where T = permutedims(mask, (2, 1, 3)) # swap i and j in [N, N, B]


"""
Specialised version for parameters or states that are <:AbstractFloat. Specific 
behaviour here is to recognize x == _safe_inf(Float32) to change into _safe_inf(T).

This is a little hacky, and can result in unexpected results for other users.
"""
function Lux.Adapt.adapt_storage(
    ::Lux.LuxEltypeAdaptor{T}, x::AbstractFloat
) where {T<:AbstractFloat}
    if x == _safe_inf(Float32)
        @warn "Detected x = _safe_inf(Float32), will convert into _safe_inf($T). If this is not intended behaviour consider overloading Adapt.adapt_storage(::LuxEltypeAdaptor{T}, x::AbstractFloat)." maxlog=1
        return _safe_inf(T)
    else
        return convert(T, x)
    end
end