"""
    triangle_attention_gpu(q, k, v, bias, mask)

Computes multi-head triangle attention optimized for GPU. Remove for loops and 
directly runs batched_mul operations over permuted arrays.
"""
function triangle_attention_gpu(
    q::AbstractArray{T, 5}, 
    k::AbstractArray{T, 5}, 
    v::AbstractArray{T, 5}, 
    bias::AbstractArray{T, 4}, 
    mask::Union{Nothing, AbstractArray{T, 3}, AbstractArray{Bool, 3}};
    neg_inf = -_safe_inf(T),
    bmul = Lux.batched_matmul
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


"""
    triangle_attention_gpu_backward!(dq, dk, dv, dbias, dout, q, k, v, bias, mask; kwargs...)

Cache-free GPU backward pass. Recomputes softmax scores on the fly and uses high-level 
array operations (batched_matmul, permutedims) to ensure compatibility with GPU libraries.
"""
triangle_attention_gpu_backward!(dq, dk, dv, dbias, dout, q, k, v, bias, mask; kwargs...) =
    _triangle_attention_gpu_backward!(dq, dk, dv, dbias, dout, q, k, v, bias, mask; kwargs...)
# The above allows us to call metal_batched_mul instead of NNlib.batched_mul for Float32


function _triangle_attention_gpu_backward!(
    dq::AbstractArray{T, 5}, dk::AbstractArray{T, 5}, dv::AbstractArray{T, 5}, dbias::AbstractArray{T, 4}, 
    dout::AbstractArray{T, 5}, 
    q::AbstractArray{T, 5}, k::AbstractArray{T, 5}, v::AbstractArray{T, 5}, bias::AbstractArray{T, 4}, 
    mask::Union{Nothing, AbstractArray{T, 3}, AbstractArray{Bool, 3}};
    neg_inf = -_safe_inf(T),
    bmul = Lux.batched_matmul
) where T
    D, H, N, _, B = size(q)
    scale = T(1.0 / sqrt(D))
    batch_size = N * H * B

    # ==========================================================
    #   Forward Recomputation
    # ==========================================================
    q_perm = copy(permutedims(q, (4, 1, 3, 2, 5))) # [N_j, D, N_i, H, B]
    q_batch_T = reshape(q_perm, N, D, batch_size)

    k_perm = copy(permutedims(k, (1, 4, 3, 2, 5))) # [D, N_j, N_i, H, B]
    k_batch = reshape(k_perm, D, N, batch_size)

    v_perm = copy(permutedims(v, (1, 4, 3, 2, 5))) 
    v_batch = reshape(v_perm, D, N, batch_size)

    scores_batch = bmul(q_batch_T, k_batch)
    scores_5d = reshape(scores_batch, N, N, N, H, B) # [N_q, N_k, N_row, H, B]
    
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

    # ==========================================================
    #   Backward Pass
    # ==========================================================
    
    dout_perm = copy(permutedims(dout, (1, 4, 3, 2, 5))) 
    dout_batch = reshape(dout_perm, D, N, batch_size)
    
    attn_batch = reshape(scores_5d, N, N, batch_size)
    dv_batch = bmul(dout_batch, attn_batch)
    
    dv_reshaped = reshape(dv_batch, D, N, N, H, B)
    dv .+= copy(permutedims(dv_reshaped, (1, 4, 3, 2, 5)))

    dout_batch_T = permutedims(dout_batch, (2, 1, 3)) # [N_q, D, batch]
    dP_batch_T = bmul(dout_batch_T, v_batch) 
    dP_5d = reshape(dP_batch_T, N, N, N, H, B)

    sum_dP_P = sum(dP_5d .* scores_5d, dims=2)
    dS_5d = scores_5d .* (dP_5d .- sum_dP_P)

    dbias_sum = sum(dS_5d, dims=3) 
    dbias_4d = reshape(dbias_sum, N, N, H, B) # [N_q, N_k, H, B]
    # Permute (N_q, N_k, H, B) -> (H, N_q, N_k, B)
    dbias .+= copy(permutedims(dbias_4d, (3, 1, 2, 4)))

    dS_5d_scaled = dS_5d .* scale
    dS_batch = reshape(dS_5d_scaled, N, N, batch_size)

    k_batch_T = permutedims(k_batch, (2, 1, 3))
    dq_batch_T = bmul(dS_batch, k_batch_T)
    dq_reshaped = reshape(dq_batch_T, N, D, N, H, B)
    dq .+= copy(permutedims(dq_reshaped, (2, 4, 3, 1, 5))) 

    q_batch_T_T = permutedims(q_batch_T, (2, 1, 3)) 
    dk_batch = bmul(q_batch_T_T, dS_batch)
    dk_reshaped = reshape(dk_batch, D, N, N, H, B)
    dk .+= copy(permutedims(dk_reshaped, (1, 4, 3, 2, 5)))

    return nothing
end

function EnzymeRules.augmented_primal(
    config::RevConfig, 
    func::Const{typeof(triangle_attention_gpu)}, 
    # Enzyme passes Duplicated if it needs the output gradient downstream
    ::Type{<:Union{Duplicated, DuplicatedNoNeed}}, 
    q, k, v, bias, mask
)
    primal_out = func.val(q.val, k.val, v.val, bias.val, mask.val)
    
    # Because the forward function does not operate in-place, we must create a matching 
    # zero-filled array for Enzyme to use to accumulate the downstream gradients.
    shadow_out = Base.zeros(eltype(primal_out), size(primal_out))
    
    # Return: (Primal Output, Shadow Output, Tape)
    # We pass `nothing` for the tape since we do forward recomputation!
    return AugmentedReturn(primal_out, shadow_out, nothing)
end

function EnzymeRules.reverse(
    config::RevConfig, 
    func::Const{typeof(triangle_attention_gpu)}, 
    dret::Union{Duplicated, DuplicatedNoNeed}, # Contains primal_out and shadow_out
    tape, # This is `nothing`
    q, k, v, bias, mask
)
    # Extract the downstream gradient (dout) from the return type
    dout = dret.dval

    triangle_attention_gpu_backward!(
        q.dval, k.dval, v.dval, bias.dval, 
        dout, 
        q.val, k.val, v.val, bias.val, mask.val
    )
    
    return (nothing, nothing, nothing, nothing, nothing)
end