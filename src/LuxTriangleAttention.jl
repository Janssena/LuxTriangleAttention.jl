module LuxTriangleAttention

import Lux, Random

using Static

include("utils.jl")
export resolve_defaults

include("layers/primitives.jl")
export LayerNormNoBias

include("layers/glu.jl")
export GatedLinearUnit, SwiGLU

include("layers/attention.jl")
include("layers/triangle_attention.jl")
export Attention, TriangleAttention, prep_mask, prep_triangle_bias

include("layers/triangle_multiplication.jl")
include("layers/tri_mul_core.jl")
export TriangleMultiplication, TriMulCore

end
