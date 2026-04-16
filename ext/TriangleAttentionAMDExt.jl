module TriangleAttentionAMDExt

using TriangleAttention: TriangleAttention, triangle_attention_gpu
using AMDGPU

TriangleAttention.triangle_attention(
    q::ROCArray{T, 5}, k::ROCArray{T, 5}, v::ROCArray{T, 5}, 
    bias::ROCArray{T, 4}, mask::Union{Nothing, ROCArray{T, 3}} = nothing
) where T = TriangleAttention.triangle_attention_gpu(q, k, v, bias, mask)

end