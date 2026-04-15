LayerNormNoBias(args...; kwargs...) = 
    Lux.Experimental.freeze(LayerNorm(args...; affine=true, init_bias=Lux.zeros32, kwargs...), (:bias, ))