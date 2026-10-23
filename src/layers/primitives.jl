const LUX_LAYERNORM = Union{Lux.LayerNorm,Lux.Experimental.FrozenLayer{<:Any,<:Lux.LayerNorm}}

"""
    StableLayerNorm(shape, activation=identity; epsilon=1f-5, dims=Colon(), affine=True(), kwargs...)
    StableLayerNorm(layer)    # a Lux.LayerNorm, or one frozen with `Lux.Experimental.freeze`

A `Lux.LayerNorm` whose statistics are always accumulated in Float32 when the data is 16-bit.

For Float32 and Float64 (and any non-16-bit input, traced or not) it IS the wrapped `Lux.LayerNorm`: the
call is `layer(x, ps, st)`, so whatever Lux and LuxLib dispatch on for the array type — today a generic
`AbstractArray` method that traced Reactant arrays flow through, and any `AnyTracedRArray` method a later
Lux adds — is used unchanged, and the result is bitwise the layer's own.

For Float16 and BFloat16 it computes the mean and variance in Float32 ([`layernorm_f32stats`](@ref)) and
returns the result in the input's type, with the layer's own scale, bias, activation, `dims` and `epsilon`.
`Lux.LayerNorm` accumulates in the input type, where the sum of squares behind the variance overflows
Float16 at |x| ≈ 20 over a few hundred channels (σ² ≈ 2e6 on AlphaFold2's pair activations is not
representable): finite but 85–99% wrong output on pair activations with σ ≈ 60 over 32 channels.

The parameters and state are the wrapped layer's own, unnested (an `AbstractLuxWrapperLayer`), so
`StableLayerNorm(shape)` has exactly the parameter tree of `Lux.LayerNorm(shape)`: the same
initialisation, the same weight loading. [`ChannelsLast`](@ref) supports it too (for the unfrozen layer).

The wrapped layer may be a `Lux.LayerNorm` frozen with `Lux.Experimental.freeze` (that is what
[`LayerNormNoBias`](@ref) builds): its frozen parameters live in the state, and the Float32 path reads them
from there.
"""
struct StableLayerNorm{L<:LUX_LAYERNORM} <: Lux.AbstractLuxWrapperLayer{:layer}
    layer::L
end

StableLayerNorm(shape, args...; kwargs...) = 
    StableLayerNorm(Lux.LayerNorm(shape, args...; kwargs...))

Base.show(io::IO, m::StableLayerNorm) = print(io, "StableLayerNorm(", m.layer, ")")

(m::StableLayerNorm)(x::AbstractArray{T}, ps, st::NamedTuple) where T = 
    m.layer(x, ps, st)

# Traced 16-bit arrays (`AbstractArray{TracedRNumber{Float16}}`) are routed here by the Reactant extension.
(m::StableLayerNorm)(x::AbstractArray{T}, ps, st::NamedTuple) where T<:Union{Float16, BFloat16} = 
    _layernorm_f32stats(m.layer, x, ps, st), st

"""
    layernorm_f32stats(x, scale, bias, activation, dims, epsilon, ::Type{T})

Normalise `x` over `dims` with the mean and variance accumulated in Float32, in ONE pass
(σ² = E[x²] − μ²), and return the result in `T`.

For 16-bit inputs the statistics cannot stay in the input type: the sum of squares over a few hundred
channels overflows Float16 at |x| ≈ 20, and σ² itself (≈2e6 on AlphaFold2's pair activations) is not
representable. One pass is safe here although it is not for Float32 inputs: for 16-bit data
accumulated in Float32 its cancellation error stays below the input's own rounding until |μ|/σ ≈ 2¹⁴.
Only μ and 1/√(σ²+ε) are converted back, so a 16-bit `x` is normalised in `T` without a Float32 copy;
a Float32 `x` (a contraction computed wide) is normalised in Float32 and then converted. `epsilon` is
used in Float32, so it may be far below `T`'s range. `scale`/`bias` (or `nothing`) must already be
broadcast-compatible with `x`; they are narrowed to `T` if wider.
"""
function layernorm_f32stats(x::AbstractArray, scale, bias, act::F, dims, epsilon, ::Type{T}) where {F,T}
    x_f32 = Float32.(x)
    n = prod(Base.Fix1(size, x), dims) |> Float32
    mean_f32 = sum(x_f32; dims) ./ n
    var_f32 = max.(sum(abs2.(x_f32); dims) ./ n .- abs2.(mean_f32), zero(Float32))

    rstd_f32 = inv.(sqrt.(var_f32 .+ Float32(epsilon)))
    y = _normalize(x, mean_f32, rstd_f32, T)
    # A wider scale/bias (e.g. `LayerNormNoBias`'s frozen bias, kept in the STATE) must not promote
    # the result out of `T`.
    isnothing(scale) || (y = y .* _narrow(scale, T))
    isnothing(bias) || (y = y .+ _narrow(bias, T))

    return act.(y)
end

# The 16-bit forward of [`StableLayerNorm`](@ref): `ln`'s own scale, bias, activation, `dims` and
# `epsilon` through [`layernorm_f32stats`](@ref). A frozen layer (`LayerNormNoBias`) keeps its frozen
# parameters (the zero bias) in the state, so they are merged back into `ps` here, as `FrozenLayer`
# itself does before calling the layer it wraps.
_layernorm_f32stats(ln::Lux.LayerNorm, x::AbstractArray, ps, st) = _layernorm_f32stats(ln, x, ps)

_layernorm_f32stats(ln::Lux.Experimental.FrozenLayer, x::AbstractArray, ps, st) =
    _layernorm_f32stats(ln.layer, x, merge(ps, st.frozen_params))

function _layernorm_f32stats(ln::Lux.LayerNorm, x::AbstractArray, ps)
    scale = haskey(ps, :scale) ? ps.scale : nothing
    bias = haskey(ps, :bias) ? ps.bias : nothing
    dims = ln.dims isa Colon ? ntuple(identity, ndims(x)) : ln.dims
    return layernorm_f32stats(x, scale, bias, ln.activation, dims, ln.epsilon, realtype(eltype(x)))
end

_narrow(a, ::Type{T}) where T = realtype(eltype(a)) === T ? a : T.(a)

# Already in T: shift and scale by the narrowed statistics. Wide (the safe contraction's Float32
# result): normalise wide, then narrow. Every cast is its OWN broadcast: inside a fused one, Reactant
# applies `T(::TracedRNumber)` element by element, which does not exist (a standalone `T.(x)` lowers
# to one whole-array convert).
function _normalize(x, mean_f32, rstd_f32, ::Type{T}) where {T}
    if realtype(eltype(x)) === T
        μ = T.(mean_f32)
        rstd = T.(rstd_f32)
        return @. (x - μ) * rstd
    end
    y32 = @. (x - mean_f32) * rstd_f32
    return T.(y32)
end

"""
    LayerNormNoBias(dims; epsilon=1f-5, kwargs...)

A [`StableLayerNorm`](@ref) (a `Lux.LayerNorm`, Float16-safe) where the bias is frozen at zero. A factory
function: the result is a `StableLayerNorm` wrapping the frozen layer, with the parameter and state tree the
frozen layer has (`ps = (; scale)`, the zero bias in `st.frozen_params`). This is useful for 
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
LayerNormNoBias(args...; kwargs...) = StableLayerNorm(Lux.Experimental.freeze(
    Lux.LayerNorm(args...; affine=true, init_bias=Lux.zeros32, kwargs...),
    (:bias,)
))