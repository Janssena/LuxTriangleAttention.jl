module LuxTriangleAttentionCUDAExt

using LuxTriangleAttention: LuxTriangleAttention, triangle_attention_gpu
using CUDA

LuxTriangleAttention.triangle_attention(
    q::CuArray{T, 5}, k::CuArray{T, 5}, v::CuArray{T, 5}, 
    bias::CuArray{T, 4}, mask::Union{Nothing, CuArray{T, 3}, CuArray{Bool, 3}} = nothing
) where T = LuxTriangleAttention.triangle_attention_gpu(q, k, v, bias, mask)

#TODO: Load triton kernels
end