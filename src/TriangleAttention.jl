module TriangleAttention

import Lux
using Tullio, LoopVectorization

include("layers/primitives.jl");
export LayerNormNoBias

include("kernels/kernels.jl");
export triangle_attention
export triangle_attention_simple!, triangle_attention_tullio!
export triangle_attention_gpu!, triangle_attention_cuda!

end # module TriangleAttention
