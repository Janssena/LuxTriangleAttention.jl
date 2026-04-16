module TriangleAttentionMetalExt

import NNlib
import LinearAlgebra: mul!

using TriangleAttention: TriangleAttention, triangle_attention_gpu
using Metal

# Patch for missing Float32 implementation of NNlib batched_mul for Metal.jl
function metal_batched_mul(A::MtlArray{Float32, 3}, B::MtlArray{Float32, 3})
    M, _, B_dim = size(A)
    _, N, _ = size(B)
    
    C = similar(A, M, N, B_dim)
    
    for i in 1:B_dim
        mul!(@view(C[:, :, i]), @view(A[:, :, i]), @view(B[:, :, i]))
    end
    
    return C
end

# Fallback to standard NNlib implementation for all other types
metal_batched_mul(A, B) = NNlib.batched_mul(A, B)

TriangleAttention.triangle_attention(
    q::MtlArray{T, 5}, k::MtlArray{T, 5}, v::MtlArray{T, 5}, 
    bias::MtlArray{T, 4}, mask::Union{Nothing, MtlArray{T, 3}, MtlArray{Bool, 3}} = nothing
) where T = triangle_attention_gpu(q, k, v, bias, mask; bmul = metal_batched_mul)

end