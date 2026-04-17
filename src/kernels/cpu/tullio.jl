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
        Threads.@threads for h in 1:H
            scores = Matrix{T}(undef, N, N)
            exp_scores = Matrix{T}(undef, N, N)
            max_workspace = Matrix{T}(undef, N, 1)
            sum_workspace = Matrix{T}(undef, N, 1)
            
            for i in 1:N
                q_i = @view q[:, h, i, :, b] 
                k_i = @view k[:, h, i, :, b]
                v_i = @view v[:, h, i, :, b]
                out_i = @view out[:, h, i, :, b]
                bias_slice = @view bias[h, :, :, b]
                
                # 1. Matmul Q * K^T
                _tullio_qk!(scores, q_i, k_i, scale)
                
                # 2. Add Bias
                @. scores += bias_slice

                # 3. Apply Mask and Compute Softmax
                _apply_mask!(scores, mask, (i, :, b); kwargs...)
                _update_exp_scores!(exp_scores, max_workspace, sum_workspace, scores, mask, (i, :, b))

                # 4. Matmul Softmax * V^T
                _tullio_out!(out_i, v_i, exp_scores)
            end
        end
    end
    
    return nothing
end

_tullio_qk!(scores, q_i, k_i, scale::T) where T<:AbstractFloat = 
    @tullio threads=false scores[j, k] = q_i[d, j] * k_i[d, k] * scale

_tullio_qk!(scores, q_i, k_i, scale::Float16) = 
    @tullio threads=false avx=false scores[j, k] = q_i[d, j] * k_i[d, k] * scale

_tullio_out!(out_i, v_i, exp_scores::AbstractMatrix{T}) where T<:AbstractFloat = 
    @tullio threads=false out_i[d, j] = v_i[d, k] * exp_scores[j, k]

_tullio_out!(out_i, v_i, exp_scores::AbstractMatrix{Float16}) = 
    @tullio threads=false avx=false out_i[d, j] = v_i[d, k] * exp_scores[j, k]


"""
    triangle_attention_tullio_backward!(dq, dk, dv, dbias, dout, q, k, v, bias, mask; kwargs...)

@tullio optimized backward pass. Recomputes Softmax blocks on the fly.
Workspaces are allocated per-thread to prevent data races.
"""
function triangle_attention_tullio_backward!(
    dq::AbstractArray{T, 5}, dk::AbstractArray{T, 5}, dv::AbstractArray{T, 5}, dbias::AbstractArray{T, 4}, 
    dout::AbstractArray{T, 5}, 
    q::AbstractArray{T, 5}, k::AbstractArray{T, 5}, v::AbstractArray{T, 5}, bias::AbstractArray{T, 4}, 
    mask::Union{Nothing, AbstractArray{T, 3}, AbstractArray{Bool, 3}}; 
    kwargs...
) where T
    D, H, N, _, B = size(q)
    scale = T(1.0 / sqrt(D))

    for b in 1:B
        Threads.@threads for h in 1:H
            # Thread-local workspaces MUST be inside the thread loop!
            scores = Matrix{T}(undef, N, N)
            exp_scores = Matrix{T}(undef, N, N)
            dP = Matrix{T}(undef, N, N)
            dS = Matrix{T}(undef, N, N)
            sum_dp = Matrix{T}(undef, N, 1)
            max_workspace = Matrix{T}(undef, N, 1)
            sum_workspace = Matrix{T}(undef, N, 1)
            
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
                # 1. Forward Recomputation
                # ==========================================================
                _tullio_qk!(scores, q_i, k_i, scale)
                @. scores += bias_slice
                
                _apply_mask!(scores, mask, (i, :, b); kwargs...)
                _update_exp_scores!(exp_scores, max_workspace, sum_workspace, scores, mask, (i, :, b))

                # ==========================================================
                # 2. Backward Pass Math (@tullio)
                # ==========================================================
                _tullio_dv!(dv_i, dout_i, exp_scores)
                
                _tullio_dp!(dP, dout_i, v_i)
                
                # Softmax Gradient
                @. dS = exp_scores * dP
                sum!(sum_dp, dS)
                @. dS = exp_scores * (dP - sum_dp)
                
                # dBias
                @. dbias_slice += dS
                
                # Scale dS for Q and K gradients
                @. dS = dS * scale
                
                _tullio_dq!(dq_i, k_i, dS)
                _tullio_dk!(dk_i, q_i, dS)
            end
        end
    end
    
    return nothing
end

# --- dV = dO * P (Accumulate) ---
_tullio_dv!(dv_i, dout_i, exp_scores::AbstractMatrix{T}) where T<:AbstractFloat = 
    @tullio threads=false dv_i[d, k] += dout_i[d, j] * exp_scores[j, k]

_tullio_dv!(dv_i, dout_i, exp_scores::AbstractMatrix{Float16}) = 
    @tullio threads=false avx=false dv_i[d, k] += dout_i[d, j] * exp_scores[j, k]

# --- dP = dO^T * V (Overwrite) ---
_tullio_dp!(dP, dout_i, v_i::AbstractMatrix{T}) where T<:AbstractFloat = 
    @tullio threads=false dP[j, k] = dout_i[d, j] * v_i[d, k]

_tullio_dp!(dP, dout_i, v_i::AbstractMatrix{Float16}) = 
    @tullio threads=false avx=false dP[j, k] = dout_i[d, j] * v_i[d, k]

# --- dQ = K * dS^T (Accumulate) ---
_tullio_dq!(dq_i, k_i, dS::AbstractMatrix{T}) where T<:AbstractFloat = 
    @tullio threads=false dq_i[d, j] += k_i[d, k] * dS[j, k]

_tullio_dq!(dq_i, k_i, dS::AbstractMatrix{Float16}) = 
    @tullio threads=false avx=false dq_i[d, j] += k_i[d, k] * dS[j, k]

# --- dK = Q * dS (Accumulate) ---
_tullio_dk!(dk_i, q_i, dS::AbstractMatrix{T}) where T<:AbstractFloat = 
    @tullio threads=false dk_i[d, k] += q_i[d, j] * dS[j, k]

_tullio_dk!(dk_i, q_i, dS::AbstractMatrix{Float16}) = 
    @tullio threads=false avx=false dk_i[d, k] += q_i[d, j] * dS[j, k]