# CPU kernels
include("cpu/amx.jl")
include("cpu/simple.jl")
include("cpu/tullio.jl")
include("gpu/generic.jl")

"""
    triangle_attention(q, k, v, bias, mask=nothing)

Computes multi-head triangle attention. Automatically routes to the fastest 
backend based on the host CPU/GPU architecture.

When set, the mask should either be the same float type as q, v, k, and bias
or be an Array of Booleans. In both cases, 1 indicates attention while 0 
results in masking at that index.

# Shapes
- `q`, `k`, `v`: `[D, H, N, N, B]`
- `bias`: `[H, N, N, B]` 
- `mask`: `[N, N, B]`
- `Returns`: `[D, H, N, N, B]`
"""
function triangle_attention end

# TODO: We can potentially think about storing some variables in a cache output 
# that we discard when calling triangle_attention, but collect in the specific 
# call to the _backward! function. This way, we don't have to recompute the 
# forward pass components in the backward function. Enzyme calls the forward
# anyway, so currently we are doing 2x forward for one backward. This might be 
# Difficult for looped versions (the cpu versions) since the exp_scores are 
# different based on the indexes we are currently working with.
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
    @. scores = ifelse(mask_slice, scores, neg_inf)
    return nothing
end

function _update_exp_scores!(exp_scores::AbstractArray{T}, max_workspace, sum_workspace, scores, mask, idxs::Tuple) where T
    maximum!(max_workspace, scores)
    @. exp_scores = exp(scores - max_workspace)
    _apply_scores_mask!(exp_scores, mask, idxs)
    sum!(sum_workspace, exp_scores)
    @. exp_scores = ifelse(sum_workspace > zero(T), exp_scores / sum_workspace, zero(T))

    return nothing
end

_apply_scores_mask!(_, ::Nothing, idxs) = nothing
function _apply_scores_mask!(exp_scores, mask, idxs) 
    mask_slice = transpose(view(mask, idxs...))
    @. exp_scores = exp_scores * mask_slice
    return nothing
end