"""
    triangle_attention_simple!(out, q, k, v, bias, mask=nothing)

Computes multi-head triangle attention for pair representations. 
Should only be used for benchmarking purposes.

# Shapes
- `q`, `k`, `v`: `[D, H, N, N, B]`
- `bias`: `[H, N, N, B]` 
- `mask`: `[N, N, B]`
- `Returns`: `[D, H, N, N, B]`
"""
function triangle_attention_simple!(
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
        for h in 1:H
            for i in 1:N
                q_i = @view q[:, h, i, :, b] 
                k_i = @view k[:, h, i, :, b]
                v_i = @view v[:, h, i, :, b]
                
                scores = transpose(q_i) * k_i
                scores .*= scale
                
                bias_slice = @view bias[h, :, :, b]
                scores .+= bias_slice
                
                _apply_mask!(scores, mask, (i, :, b))

                max_scores = maximum(scores, dims=2)
                exp_scores = exp.(scores .- max_scores)
                attn_weights = exp_scores ./ sum(exp_scores, dims=2)
            
                out_i = v_i * transpose(attn_weights)
                out[:, h, i, :, b] .= out_i
            end
        end
    end
    
    return nothing
end