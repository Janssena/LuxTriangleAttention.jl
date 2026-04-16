include("cpu/simple.jl")
include("cpu/tullio.jl")
include("cpu/amx.jl")
include("gpu/generic.jl")
include("gpu/cuda.jl")

"""
    triangle_attention(q, k, v, bias, mask=nothing)

Computes multi-head triangle attention. 
Automatically routes to the fastest backend based on the host CPU architecture.
"""
function triangle_attention(q, k, v, pair, mask=nothing)
    out = similar(q)
    triangle_attention!(out, q, k, v, pair, mask)
    return out
end

function triangle_attention!(out::AbstractArray, q, k, v, bias, mask=nothing)
    if IS_APPLE_SILICON
        triangle_attention_amx!(out, q, k, v, bias, mask)
    else
        triangle_attention_tullio!(out, q, k, v, bias, mask)
    end
end

# triangle_attention!(out::CuArray, args...) = 
#     triangle_attention_cuda!(out, args...)

##### Helpers
_safe_inf(::Type{T}) where T<:AbstractFloat = T(1e9)
_safe_inf(::Type{Float16}) = Float16(1e4)

_apply_mask!(_, ::Nothing, idxs) = nothing
function _apply_mask!(scores::AbstractArray{T, 2}, mask::AbstractArray{T, 3}, idxs::Tuple; neg_inf=-_safe_inf(T)) where T
    mask_slice = transpose(view(mask, idxs...))
    @. scores = ifelse(mask_slice == 0, neg_inf, scores)
    return nothing
end