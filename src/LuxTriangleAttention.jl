module LuxTriangleAttention

const IS_APPLE_SILICON = Sys.isapple() && Sys.ARCH === :aarch64

import Lux, Random

using Tullio, LoopVectorization, Enzyme, Enzyme.EnzymeRules, Static, PrecompileTools

include("layers/glu.jl");
export GatedLinearUnit, SwiGLU

include("kernels/kernels.jl");
export triangle_attention
export triangle_attention_simple!, triangle_attention_tullio!, triangle_attention_amx!
export triangle_attention_amx_backward!, triangle_attention_tullio_backward!, triangle_attention_gpu_backward!
export triangle_attention_gpu

include("layers/triangle_attention.jl");
include("layers/tri_attn_core.jl");
export TriangleAttention, TriAttnCore

include("layers/triangle_multiplication.jl");
include("layers/tri_mul_core.jl");
export TriangleMultiplication, TriMulCore

# Precompile attention kernels
@setup_workload begin
    D, H, N, B = 4, 2, 8, 1
    tri_attn_fn! = IS_APPLE_SILICON ? triangle_attention_amx! : triangle_attention_tullio!
    tri_attn_fn_backward! = IS_APPLE_SILICON ? triangle_attention_amx_backward! : triangle_attention_tullio_backward!
    # We want to precompile for Float32 since that is standard for ML
    for T in [Float16, Float32]
        q = rand(T, D, H, N, N, B)
        k = rand(T, D, H, N, N, B)
        v = rand(T, D, H, N, N, B)
        bias = rand(T, H, N, N, B)
        
        # We precompile both a "No Mask" and a "Bool Mask" version
        mask_none = nothing
        mask_bool = rand(Bool, N, N, B)
        mask_float = rand(T[0, 1], N, N, B)
    
        out = zeros(T, size(q))
        dout = ones(T, size(out))
        
        dq = zeros(T, size(q))
        dk = zeros(T, size(k))
        dv = zeros(T, size(v))
        dbias = zeros(T, size(bias))
        
        @compile_workload begin
            for _mask in [mask_none, mask_bool, mask_float]
                tri_attn_fn!(out, q, k, v, bias, _mask)
                tri_attn_fn_backward!(dq, dk, dv, dbias, dout, q, k, v, bias, _mask)
            end
        end
    end
end

end