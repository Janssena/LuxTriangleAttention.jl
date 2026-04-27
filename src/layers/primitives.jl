"""
    LayerNormNoBias(dims; epsilon=1f-5, kwargs...)

A variant of `Lux.LayerNorm` where the bias is frozen at zero. This is useful for architectures
where affine transformation is desired but without a learnable bias term.
"""
LayerNormNoBias(args...; kwargs...) = Lux.Experimental.freeze(
    Lux.LayerNorm(args...; affine=true, init_bias=Lux.zeros32, kwargs...),
    (:bias,)
)