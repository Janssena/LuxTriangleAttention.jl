import LinearAlgebra: mul!, transpose

"""
    triangle_attention_amx!(out, q, k, v, bias, mask)

Computes multi-head triangle attention for pair representations. Uses mul! 
instead of tullio to leverage the faster Accelerate backend on Apple silicon.
"""
# --- Main Kernel ---
function triangle_attention_amx!(
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
        
        for h in 1:H
            for i in 1:N
                q_i = @view q[:, h, i, :, b] 
                k_i = @view k[:, h, i, :, b]
                v_i = @view v[:, h, i, :, b]
                out_i = @view out[:, h, i, :, b]
                bias_slice = @view bias[h, :, :, b]
                
                mul!(scores, transpose(q_i), k_i)
                @. scores = (scores * scale) + bias_slice
                
                _apply_mask!(scores, mask, (i, :, b); kwargs...)
                _update_exp_scores!(exp_scores, max_workspace, sum_workspace, scores, mask, (i, :, b))

                mul!(out_i, v_i, transpose(exp_scores))
            end
        end
    end
    
    return nothing
end

"""
    triangle_attention_amx_backward!(dq, dk, dv, dbias, dout, q, k, v, bias, mask; kwargs...)

AMX-optimized backward pass. Recomputes the Softmax blocks on the fly to save VRAM and 
leverages `mul!` for hardware acceleration.
"""
function triangle_attention_amx_backward!(
    dq::AbstractArray{T, 5}, dk::AbstractArray{T, 5}, dv::AbstractArray{T, 5}, dbias::AbstractArray{T, 4}, 
    dout::AbstractArray{T, 5}, 
    q::AbstractArray{T, 5}, k::AbstractArray{T, 5}, v::AbstractArray{T, 5}, bias::AbstractArray{T, 4}, 
    mask::Union{Nothing, AbstractArray{T, 3}, AbstractArray{Bool, 3}}; 
    kwargs...
) where T
D, H, N, _, B = size(q)
    scale = T(1.0 / sqrt(D))

    for b in 1:B
        # Thread-local workspaces
        scores = Matrix{T}(undef, N, N)
        exp_scores = Matrix{T}(undef, N, N)
        dP = Matrix{T}(undef, N, N)
        dS = Matrix{T}(undef, N, N)
        sum_dp = Matrix{T}(undef, N, 1)
        max_workspace = Matrix{T}(undef, N, 1)
        sum_workspace = Matrix{T}(undef, N, 1)
        
        for h in 1:H
            for i in 1:N
                # Forward Primal Views
                q_i = @view q[:, h, i, :, b] 
                k_i = @view k[:, h, i, :, b]
                v_i = @view v[:, h, i, :, b]
                bias_slice = @view bias[h, :, :, b]
                
                # Gradient Views
                dout_i = @view dout[:, h, i, :, b]
                dq_i = @view dq[:, h, i, :, b]
                dk_i = @view dk[:, h, i, :, b]
                dv_i = @view dv[:, h, i, :, b]
                dbias_slice = @view dbias[h, :, :, b]
                
                # ==========================================================
                #   Forward recomputation
                # ==========================================================
                mul!(scores, transpose(q_i), k_i)
                @. scores = (scores * scale) + bias_slice
                
                _apply_mask!(scores, mask, (i, :, b); kwargs...)
                _update_exp_scores!(exp_scores, max_workspace, sum_workspace, scores, mask, (i, :, b))

                # ==========================================================
                #   Backward Pass Math
                # ==========================================================
                # dV = dO * P  (Accumulate into dv_i)
                mul!(dv_i, dout_i, exp_scores, one(T), one(T))
                # dP = dO^T * V
                mul!(dP, transpose(dout_i), v_i)
                # Softmax Gradient: dS = P .* (dP - sum(P .* dP, dims=2))
                @. dS = exp_scores * dP
                sum!(sum_dp, dS)
                @. dS = exp_scores * (dP - sum_dp)
                # dBias = dS (Accumulate)
                @. dbias_slice += dS
                # Scale dS for dQ and dK
                @. dS = dS * scale
                # dQ = K * dS^T (Accumulate into dq_i)
                mul!(dq_i, k_i, transpose(dS), one(T), one(T))
                # dK = Q * dS (Accumulate into dk_i)
                mul!(dk_i, q_i, dS, one(T), one(T))
            end
        end
    end
    
    return nothing
end

# The Forward Hook, TODO: think about collecting the exp_scores here
function EnzymeRules.augmented_primal(
    config::RevConfig, 
    func::Const{typeof(triangle_attention_amx!)}, 
    ret_type::Type{<:Const}, 
    out, q, k, v, bias, mask
)
    # Execute the actual forward pass
    func.val(out.val, q.val, k.val, v.val, bias.val, mask.val)
    
    # Return an empty tape since we recompute exp_scores in the backward pass
    return AugmentedReturn(nothing, nothing, nothing)
end

# The Backward Hook
function EnzymeRules.reverse(
    config::RevConfig, 
    func::Const{typeof(triangle_attention_amx!)}, 
    ret_type::Type{<:Const}, 
    tape, 
    out, q, k, v, bias, mask
)
    # Check if a gradient is actively being requested for the output
    if out.dval !== nothing
        triangle_attention_amx_backward!(
            q.dval, k.dval, v.dval, bias.dval, 
            out.dval, 
            q.val, k.val, v.val, bias.val, mask.val
        )
    end
    
    # Return a tuple of `nothing` matching the number of arguments in the signature 
    # (out, q, k, v, bias, mask). This tells Enzyme the rule successfully executed.
    return (nothing, nothing, nothing, nothing, nothing, nothing)
end