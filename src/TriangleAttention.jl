module TriangleAttention

const IS_APPLE_SILICON = Sys.isapple() && Sys.ARCH === :aarch64

import Lux
using Tullio, LoopVectorization, NNlib

include("layers/primitives.jl");
export LayerNormNoBias

include("kernels/kernels.jl");
export triangle_attention
export triangle_attention_simple!, triangle_attention_tullio!, triangle_attention_amx!
export triangle_attention_gpu

end
