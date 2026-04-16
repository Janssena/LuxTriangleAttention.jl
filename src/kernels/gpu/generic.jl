"""
    triangle_attention_gpu(q, k, v, bias, mask)

Computes multi-head triangle attention optimized for GPU.
"""
function triangle_attention_gpu(
    q::AbstractArray{T, 5}, 
    k::AbstractArray{T, 5}, 
    v::AbstractArray{T, 5}, 
    bias::AbstractArray{T, 4}, 
    mask::Union{Nothing, AbstractArray{T, 3}};
    neg_inf = -_safe_inf(T),
    bmul = NNlib.batched_mul
) where T
    D, H, N, _, B = size(q)
    scale = eltype(q)(1.0 / sqrt(D))
    batch_size = N * H * B

    # Original q: [D, H, N_i, N_j, B]
    # We want q_batch to act like it was transposed: [N_j, D, batch_size]
    # So we pull N_j (dim 4) to the front, then D (dim 1).
    q_perm = permutedims(q, (4, 1, 3, 2, 5))
    q_batch_T = reshape(q_perm, N, D, batch_size)

    k_perm = permutedims(k, (1, 4, 3, 2, 5))
    k_batch = reshape(k_perm, D, N, batch_size)

    v_perm = permutedims(v, (1, 4, 3, 2, 5))
    v_batch = reshape(v_perm, D, N, batch_size)

    scores_batch = bmul(q_batch_T, k_batch)

    scores_5d = reshape(scores_batch, N, N, N, H, B)
    bias_reshaped = reshape(permutedims(bias, (2, 3, 1, 4)), N, N, 1, H, B)

    if isnothing(mask)
        @. scores_5d = (scores_5d * scale) + bias_reshaped
    else
        mask_reshaped = reshape(permutedims(mask, (2, 1, 3)), 1, N, N, 1, B)

        @. scores_5d = ifelse(
            iszero(mask_reshaped), 
            neg_inf, 
            (scores_5d * scale) + bias_reshaped
        )
    end

    NNlib.softmax!(scores_5d, scores_5d, dims=2)

    attn_batch = reshape(scores_5d, N, N, batch_size)    
    attn_batch_T = permutedims(attn_batch, (2, 1, 3))
    out_batch = bmul(v_batch, attn_batch_T)

    out_perm = reshape(out_batch, D, N, N, H, B)
    return permutedims(out_perm, (1, 4, 3, 2, 5))
end

# function triangle_attention_gpu(
#     q::AbstractArray{T, 5}, 
#     k::AbstractArray{T, 5}, 
#     v::AbstractArray{T, 5}, 
#     bias::AbstractArray{T, 4}, 
#     mask::Union{Nothing, AbstractArray{T, 3}};
#     neg_inf = -_safe_inf(T)
# ) where T
#     D, H, N, _, B = size(q)
#     scale = T(1.0 / sqrt(D))

#     # 1. Permute
#     q_perm = permutedims(q, (1, 4, 3, 2, 5))
#     k_perm = permutedims(k, (1, 4, 3, 2, 5))
#     v_perm = permutedims(v, (1, 4, 3, 2, 5))

#     # 2. Reshape to Batched 3D
#     batch_size = N * H * B
#     q_batch = reshape(q_perm, D, N, batch_size)
#     k_batch = reshape(k_perm, D, N, batch_size)
#     v_batch = reshape(v_perm, D, N, batch_size)

#     # 3. Q * K^T (Allocates a new array in VRAM)
#     scores_batch = batched_mul(batched_transpose(q_batch), k_batch)

#     # Reshape views (Zero allocations)
#     scores_5d = reshape(scores_batch, N, N, N, H, B)
#     bias_perm = permutedims(bias, (2, 3, 1, 4))
#     bias_reshaped = reshape(bias_perm, N, N, 1, H, B)

#     # 4. FUSED BROADCAST KERNEL (Massive speedup)
#     if isnothing(mask)
#         # Apply scale and bias in one pass
#         @. scores_5d = (scores_5d * scale) + bias_reshaped
#     else
#         mask_perm = permutedims(mask, (2, 1, 3))
#         mask_reshaped = reshape(mask_perm, 1, N, N, 1, B)
        
#         # Apply mask, scale, and bias in ONE single pass
#         # Notice `zero(T)` fixes the 64-bit Metal crash
#         @. scores_5d = ifelse(
#             iszero(mask_reshaped), 
#             neg_inf, 
#             (scores_5d * scale) + bias_reshaped
#         )
#     end

#     # 5. In-Place Softmax (Saves a massive VRAM allocation)
#     NNlib.softmax!(scores_5d, scores_5d, dims=2)
    
#     # 6. Final Matmul
#     attn_batch = reshape(scores_5d, N, N, batch_size)
#     out_batch = batched_mul(v_batch, batched_transpose(attn_batch))

#     # 7. Reconstruct
#     out_perm = reshape(out_batch, D, N, N, H, B)
#     out = permutedims(out_perm, (1, 4, 3, 2, 5))

#     return out
# end