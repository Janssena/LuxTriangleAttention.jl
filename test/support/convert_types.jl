# `Lux.f64`/`f32`/`f16` by element type: moves a `Lux.setup` result to `T`. Used by the layer tests, with and
# without Python, so it lives here rather than in `test/python/setup.jl` (which includes this file for the
# benchmark scripts, which load that one on its own).

convert_types(::Type{Float64}) = Lux.f64
convert_types(::Type{Float32}) = Lux.f32
convert_types(::Type{Float16}) = Lux.f16
