include("cpu/simple.jl")
include("cpu/tullio.jl")
include("gpu/generic.jl")
include("gpu/cuda.jl")

function triangle_attention(q, k, v, pair, mask=nothing)
    out = similar(q)
    triangle_attention!(out, q, k, v, pair, mask)
    return out
end

triangle_attention!(out::AbstractArray, args...) = 
    triangle_attention_tullio!(out, args...)

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