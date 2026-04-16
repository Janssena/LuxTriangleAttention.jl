# CPU kernels
include("cpu/amx.jl")
include("cpu/simple.jl")
include("cpu/tullio.jl")
include("gpu/generic.jl")

"""
    triangle_attention(q, k, v, bias, mask=nothing)

Computes multi-head triangle attention. Automatically routes to the fastest 
backend based on the host CPU/GPU architecture.
"""
function triangle_attention end

function triangle_attention(q::Array, k, v, pair, mask=nothing; kwargs...)
    out = similar(q)
    triangle_attention!(out, q, k, v, pair, mask; kwargs...)
    return out
end

function triangle_attention!(out::Array, args...; kwargs...)
    if IS_APPLE_SILICON
        triangle_attention_amx!(out, args...; kwargs...)
    else
        triangle_attention_tullio!(out, args...; kwargs...)
    end
end
    
##### Helpers

_safe_inf(::Type{T}) where T<:AbstractFloat = T(1e9)
_safe_inf(::Type{Float16}) = Float16(1e4)

_apply_mask!(_, ::Nothing, idxs) = nothing
function _apply_mask!(scores::AbstractArray{T, 2}, mask::AbstractArray{T, 3}, idxs::Tuple; neg_inf=-_safe_inf(T)) where T
    mask_slice = transpose(view(mask, idxs...))
    @. scores = ifelse(mask_slice == zero(T), neg_inf, scores)
    return nothing
end

# TODO: Should we be using AbstractArray{Bool} for masks?
function _apply_mask!(scores::AbstractArray{T, 2}, mask::AbstractArray{Bool, 3}, idxs::Tuple; neg_inf=-_safe_inf(T)) where T
    mask_slice = transpose(view(mask, idxs...))
    @. scores = ifelse(mask_slice, neg_inf, scores)
    return nothing
end