"""
    triangle_attention_tullio!(out, q, k, v, bias, mask)

Computes multi-head triangle attention for pair representations. 
Uses @tullio and @inbounds to generatie fast and efficient loops.

# Shapes
- `q`, `k`, `v`: `[D, H, N, N, B]`
- `bias`: `[H, N, N, B]` 
- `mask`: `[N, N, B]`
- `Returns`: `[D, H, N, N, B]`
"""
# --- Main Kernel ---
function triangle_attention_tullio!(
    out::AbstractArray{T, 5}, 
    q::AbstractArray{T, 5}, 
    k::AbstractArray{T, 5}, 
    v::AbstractArray{T, 5}, 
    bias::AbstractArray{T, 4}, 
    mask::Union{Nothing, AbstractArray{T, 3}, AbstractArray{Bool, 3}}; 
    kwargs...
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
                
                _tullio_qk!(scores, q_i, k_i, scale)
                @. scores += bias_slice

                _apply_mask!(scores, mask, (i, :, b); kwargs...)

                maximum!(max_workspace, scores)
                @. exp_scores = exp(scores - max_workspace)
                sum!(sum_workspace, exp_scores)
                @. exp_scores = exp_scores / sum_workspace

                _tullio_out!(out_i, v_i, exp_scores)
            end
        end
    end
    
    return nothing
end

_tullio_qk!(scores, q_i, k_i, scale::T) where T = 
    @tullio threads=false scores[j, k] = q_i[d, j] * k_i[d, k] * scale

_tullio_qk!(scores, q_i, k_i, scale::Float16) = 
    @tullio threads=false avx=false scores[j, k] = q_i[d, j] * k_i[d, k] * scale

_tullio_out!(out_i, v_i, exp_scores::AbstractMatrix{T}) where T = 
    @tullio threads=false out_i[d, j] = v_i[d, k] * exp_scores[j, k]

_tullio_out!(out_i, v_i, exp_scores::AbstractMatrix{Float16}) = 
    @tullio threads=false avx=false out_i[d, j] = v_i[d, k] * exp_scores[j, k]