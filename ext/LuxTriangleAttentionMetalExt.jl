module LuxTriangleAttentionMetalExt

import Lux
import LinearAlgebra: mul!

using LuxTriangleAttention: LuxTriangleAttention, triangle_attention_gpu
using Metal, Enzyme, Enzyme.EnzymeRules

# Patch for missing Float32 implementation of batched_matmul for Metal.jl
function metal_batched_mul(A::MtlArray{Float32, 3}, B::MtlArray{Float32, 3})
    M, _, B_dim = size(A)
    _, N, _ = size(B)
    
    C = similar(A, M, N, B_dim)
    
    for i in 1:B_dim
        mul!(@view(C[:, :, i]), @view(A[:, :, i]), @view(B[:, :, i]))
    end
    
    return C
end

# Fallback to standard Lux.batched_matmul implementation for all other types
metal_batched_mul(A, B) = Lux.batched_matmul(A, B)

LuxTriangleAttention.triangle_attention(
    q::MtlArray{T, 5}, k::MtlArray{T, 5}, v::MtlArray{T, 5}, 
    bias::MtlArray{T, 4}, mask::Union{Nothing, MtlArray{T, 3}, MtlArray{Bool, 3}} = nothing;
    kwargs...
) where T = triangle_attention_gpu(q, k, v, bias, mask; bmul = metal_batched_mul, kwargs...)

LuxTriangleAttention.triangle_attention_gpu_backward!(
    dq::MtlArray{T, 5}, dk::MtlArray{T, 5}, dv::MtlArray{T, 5}, 
    dbias::MtlArray{T, 4}, dout::MtlArray{T, 5}, 
    q::MtlArray{T, 5}, k::MtlArray{T, 5}, v::MtlArray{T, 5}, 
    bias::MtlArray{T, 4}, mask::Union{Nothing, MtlArray{T, 3}, MtlArray{Bool, 3}}; 
    kwargs...
) where T = _triangle_attention_gpu_backward!(
    dq, dk, dv, dbias, dout, q, k, v, bias, mask; 
    bmul = metal_batched_mul, kwargs...
)

end