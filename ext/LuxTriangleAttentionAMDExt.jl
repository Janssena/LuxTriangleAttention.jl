module LuxTriangleAttentionAMDExt

using LuxTriangleAttention: LuxTriangleAttention, triangle_attention_gpu
using AMDGPU

LuxTriangleAttention.triangle_attention(
    q::ROCArray{T, 5}, k::ROCArray{T, 5}, v::ROCArray{T, 5}, 
    bias::ROCArray{T, 4}, mask::Union{Nothing, ROCArray{T, 3}, ROCArray{Bool, 3}} = nothing
) where T = LuxTriangleAttention.triangle_attention_gpu(q, k, v, bias, mask)

#TODO: Load triton kernels
end