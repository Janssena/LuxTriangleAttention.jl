"""
    LayerNormNoBias(args...; kwargs...)

There is no standard way of setting use_bias = false for Lux.LayerNorm
so we instead freeze the bias and init with zeros to achieve the same effect.
"""
LayerNormNoBias(args...; kwargs...) = Lux.Experimental.freeze(
    LayerNorm(args...; affine=true, init_bias=Lux.zeros32, kwargs...), 
    (:bias, )
)