@doc raw"""
    TriangleMultiplication(chn_in, chn_hidden; is_outgoing=true, kwargs...)

Implementation of Triangle Multiplication (AlphaFold2 Algorithms 11 & 12).
Updates the pair representation by combining edges sharing a common node. AlphaFold2 Algorithm 11
(outgoing edges) is

```math
\begin{aligned}
z_{ij} &\leftarrow \mathrm{LayerNorm}(z_{ij}) \\
a_{ij},\, b_{ij} &= \mathrm{sigmoid}\big(\mathrm{Linear}(z_{ij})\big) \odot \mathrm{Linear}(z_{ij}) \\
g_{ij} &= \mathrm{sigmoid}\big(\mathrm{Linear}(z_{ij})\big) \\
\tilde{z}_{ij} &= g_{ij} \odot \mathrm{Linear}\Big(\mathrm{LayerNorm}\big(\textstyle\sum_k a_{ik} \odot b_{jk}\big)\Big)
\end{aligned}
```

and AlphaFold2 Algorithm 12 (incoming edges) sums ``a_{ki} \odot b_{kj}`` instead.

## Arguments
- `chn_in`: Input channels.
- `chn_hidden`: Hidden channels for internal projections.

## Keyword Arguments
- `is_outgoing`: If `true`, performs "Outgoing" multiplication (AlphaFold2 Algorithm 11).
  If `false`, performs "Incoming" multiplication (AlphaFold2 Algorithm 12).
- `layernorm_eps`: Epsilon for LayerNorm.
- `layer_norm_bias`: If `false`, both LayerNorms (input and output) have a scale but no bias. (default: `true`)
- `rank`: The rank (ndims) of the input.
- `use_kernel`: If `true` (or `static(true)`), device arrays run the layer through the [`TriMul`](@ref) op, whose
  `:default` kernel is a staged, memory-lean forward and backward: about 2–3× faster and 5× less peak memory than the plain
  route on an Apple M1 Pro. It covers Float32, Float16 and BFloat16 data (parameters of the same type, every
  [`ContractionMode`](@ref)) with sigmoid gates; its backward covers Float32. Anything else runs the plain forward, and so
  does a Reactant trace (the traced layer is the faster program there). Differentiating a configuration the backward
  does not cover throws: train that one with `use_kernel=false`. Host arrays always run the plain forward.
  (default: `false`)
- `kwargs...`: Passed to `TriMulCore`.

## Inputs
- `x`: Input tensor with shape `[C, N, N, B]`.
- `mask`: Optional mask with shape `[N, N, B]`.

## Returns
- `y`: Output tensor with shape `[C, N, N, B]`.
- `st`: Updated state.
"""
struct TriangleMultiplication{LN, CORE, UK<:StaticBool} <: Lux.AbstractLuxContainerLayer{(:layer_norm, :core)}
    layer_norm::LN
    core::CORE
    use_kernel::UK
end

TriangleMultiplication(layer_norm, core) = TriangleMultiplication(layer_norm, core, static(false))


function TriangleMultiplication(
    chn_in::Int, chn_hidden::Int;
    is_outgoing::Union{Bool, StaticBool}=static(true), rank=4,
    layernorm_eps=1f-5, layer_norm_bias::Bool=true, use_kernel::Union{Bool, StaticBool}=static(false), kwargs...
)
    ln_shape = (chn_in, ntuple(one, rank-2)...)
    return TriangleMultiplication(
        layer_norm_bias ? StableLayerNorm(ln_shape; dims=1, epsilon=layernorm_eps) :
                          LayerNormNoBias(ln_shape; dims=1, epsilon=layernorm_eps),
        TriMulCore(chn_in, chn_hidden; is_outgoing, layernorm_eps, layer_norm_bias, rank, kwargs...),
        static(use_kernel)
    )
end

(m::TriangleMultiplication)(x::AbstractArray, ps, st) = m(x, nothing, ps, st)
(m::TriangleMultiplication)(inputs::NamedTuple, ps, st) = m(
    inputs.x, 
    get(inputs, :mask, nothing), 
    ps, st
)

(m::TriangleMultiplication)(x::AbstractArray, mask, ps, st) = _trimul(m.use_kernel, m, x, mask, ps, st)

# `use_kernel` routes by type: the op (its kernel on a device, this layer's forward on the host) or the plain forward.
_trimul(::True, m, x, mask, ps, st) = TriMul()(m, x, mask, ps, st)
_trimul(::False, m, x, mask, ps, st) = _trimul_layer(m, x, mask, ps, st)

function _trimul_layer(m::TriangleMultiplication, x::AbstractArray, mask, ps, st)
    x̃, layer_norm = m.layer_norm(x, ps.layer_norm, st.layer_norm)
    out, core = m.core(x̃, mask, ps.core, st.core)
    return out, merge(st, (; layer_norm, core))
end