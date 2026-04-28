"""
    LayerNormNoBias(dims; epsilon=1f-5, kwargs...)

A variant of `Lux.LayerNorm` where the bias is frozen at zero. This is useful for 
architectures where affine transformation is desired but without a learnable bias term.

## Arguments
- See Lux.LayerNorm

## Keyword Arguments
- `kwargs...`: Passed to `Lux.LayerNorm`.

## Inputs
- `x`: Input tensor.

## Returns
- `y`: Normalized tensor.
- `st`: Updated state.
"""
LayerNormNoBias(args...; kwargs...) = Lux.Experimental.freeze(
    Lux.LayerNorm(args...; affine=true, init_bias=Lux.zeros32, kwargs...),
    (:bias,)
)