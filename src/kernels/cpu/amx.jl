import LinearAlgebra: mul!, transpose

"""
    triangle_attention_amx!(out, q, k, v, bias, mask=nothing)

Computes multi-head triangle attention for pair representations. 
Uses mul! to leverage the faster Accelerate framework on Apple silicon

# Shapes
- `q`, `k`, `v`: `[D, H, N, N, B]`
- `bias`: `[H, N, N, B]` 
- `mask`: `[N, N, B]`
- `Returns`: `[D, H, N, N, B]`
"""
# --- Main Kernel ---
function triangle_attention_amx!(
    out::AbstractArray{T, 5}, 
    q::AbstractArray{T, 5}, 
    k::AbstractArray{T, 5}, 
    v::AbstractArray{T, 5}, 
    bias::AbstractArray{T, 4}, 
    mask::Union{Nothing, AbstractArray{T, 3}}
) where T
    D, H, N, _, B = size(q)
    scale = T(1.0 / sqrt(D))

    for b in 1:B
        scores = Matrix{T}(undef, N, N)
        exp_scores = Matrix{T}(undef, N, N)
        max_workspace = Matrix{T}(undef, N, 1)
        sum_workspace = Matrix{T}(undef, N, 1)
        
        Threads.@threads for h in 1:H
            for i in 1:N
                q_i = @view q[:, h, i, :, b] 
                k_i = @view k[:, h, i, :, b]
                v_i = @view v[:, h, i, :, b]
                out_i = @view out[:, h, i, :, b]
                bias_slice = @view bias[h, :, :, b]
                
                mul!(scores, transpose(q_i), k_i)
                @. scores = (scores * scale) + bias_slice
                
                _apply_mask!(scores, mask, (i, :, b))

                maximum!(max_workspace, scores)
                @. exp_scores = exp(scores - max_workspace)
                sum!(sum_workspace, exp_scores)
                @. exp_scores = exp_scores / sum_workspace

                mul!(out_i, v_i, transpose(exp_scores))
            end
        end
    end
    
    return nothing
end