"""
    ChannelsLast(layer)

`layer` applied to CHANNELS-LAST data: input `[..., in]`, output `[..., out]`, where Lux's own
convention is channels-first (`[in, ...]`). Supported: `Lux.Dense`, `Lux.LayerNorm` (normalising
the channel axis only) and [`GatedLinearUnit`](@ref)/[`SwiGLU`](@ref). Any other layer throws.

The parameters and states are the wrapped layer's own, unnested (an `AbstractLuxWrapperLayer`), so
`ChannelsLast(Dense(chn => hidden))` has exactly the parameter tree of `Dense(chn => hidden)`: the same
initialisation, the same weight loading.

## Why

A `Dense` on channels-last data computes `x · Wᵀ` instead of `W · x`, and the result is
channels-last too. Inside a layer this is often cheaper:
- a channels-last `[N², H]` projection IS the per-channel `[N, N]` "planes" layout a batched
  contraction over tokens wants (e.g. TriMulCore), so no `permutedims` is needed before it;
- under Reactant → jax-mps, channels-last chains lower to row-major `dot_general`s, which jax-mps runs
  1.1–1.9× faster (docs/FEATURE_jaxmps_dot_orientation.md §3b).

Whether it pays off is a per-layer measurement, not a rule: see test/scratch/layout_harness.jl.

## Entering and leaving the layout

A 2-D input may be a lazy `transpose` of channels-first data: `ChannelsLast(d)(transpose(x), ps, st)`
computes `xᵀ · Wᵀ` directly, with no copy. The way back is one `permutedims` at the end of the layer.

## Example

```julia
d = ChannelsLast(Lux.Dense(128 => 512))
ps, st = Lux.setup(rng, d)                 # same as for Lux.Dense(128 => 512)
y, _ = d(randn(Float32, 64, 64, 128), ps, st)   # [64, 64, 512]
```
"""
struct ChannelsLast{L<:Lux.AbstractLuxLayer} <: Lux.AbstractLuxWrapperLayer{:layer}
    layer::L
end

Base.show(io::IO, m::ChannelsLast) = print(io, "ChannelsLast(", m.layer, ")")

# The wrapper's inherited fallback would run the inner layer on channels-last data as if it were
# channels-first — silently wrong. Every supported layer has its own method below.
(m::ChannelsLast)(x, ps, st) = throw(ArgumentError("ChannelsLast does not support $(typeof(m.layer).name.name)"))

# `[..., C]` as a `[T, C]` matrix, and back. A matrix is used as it is, so a lazy `transpose` stays
# lazy and reaches the matmul as a transposed operand.
_rows(x::AbstractMatrix) = x
_rows(x::AbstractArray) = reshape(x, :, size(x, ndims(x)))
_unrows(y::AbstractMatrix, x::AbstractMatrix) = y
_unrows(y::AbstractMatrix, x::AbstractArray) = reshape(y, Base.front(size(x))..., size(y, 2))

function (m::ChannelsLast{<:Lux.Dense})(x::AbstractArray, ps, st)
    d = m.layer
    size(x, ndims(x)) == d.in_dims || throw(DimensionMismatch(
        "ChannelsLast($d) expects $(d.in_dims) channels in the LAST dimension, got size $(size(x))"))
    y = _bias_act(d.activation, _rows(x) * transpose(ps.weight), ps)
    return _unrows(y, x), st
end

_bias_act(σ, y, ps) = haskey(ps, :bias) ? σ.(y .+ transpose(ps.bias)) : σ === identity ? y : σ.(y)

# The wrapped LayerNorm's scale and bias as broadcastable against channels-last `x` (channels on the LAST
# axis), after checking it normalises the channel axis only.
function _cl_layernorm_args(ln::Lux.LayerNorm, x::AbstractArray{T,N}, ps) where {T,N}
    (ln.dims == 1 || ln.dims == (1,)) && all(isone, Base.tail(Tuple(ln.shape))) || throw(ArgumentError(
        "ChannelsLast(LayerNorm) needs a LayerNorm over the channel axis only (`LayerNorm((C,); dims=1)`), got $ln"))
    pshape = (ntuple(one, N - 1)..., size(x, N))
    scale = haskey(ps, :scale) ? reshape(ps.scale, pshape) : nothing
    bias = haskey(ps, :bias) ? reshape(ps.bias, pshape) : nothing
    return scale, bias
end

function (m::ChannelsLast{<:Lux.LayerNorm})(x::AbstractArray{T,N}, ps, st) where {T,N}
    ln = m.layer
    scale, bias = _cl_layernorm_args(ln, x, ps)
    # Uses `Lux.Utils.eltype`, not `T`: under Reactant `T` is a `TracedRNumber`.
    eps = convert(Lux.Utils.eltype(x), ln.epsilon)
    return Lux.LuxLib.layernorm(x, scale, bias, ln.activation, N, eps), st
end

# `StableLayerNorm`: the plain LayerNorm above for anything but 16-bit data; Float32 statistics for that.
function (m::ChannelsLast{<:StableLayerNorm{<:Lux.LayerNorm}})(x::AbstractArray{T,N}, ps, st) where {T,N}
    ln = m.layer.layer
    ishalf(realtype(eltype(x))) || return ChannelsLast(ln)(x, ps, st)
    scale, bias = _cl_layernorm_args(ln, x, ps)
    return layernorm_f32stats(x, scale, bias, ln.activation, N, ln.epsilon, realtype(eltype(x))), st
end

function (m::ChannelsLast{<:GatedLinearUnit{True}})(x::AbstractArray{T,N}, ps, st) where {T,N}
    glu = m.layer
    proj, _ = ChannelsLast(glu.linear)(x, ps.linear, st.linear)      # [..., 2·out]: values, then gates
    n = glu.out_chn
    front = ntuple(Returns(Colon()), N - 1)
    y = view(proj, front..., 1:n) .* glu.activation.(view(proj, front..., (n + 1):(2n)))
    return y, st
end

(m::ChannelsLast{<:GatedLinearUnit{False}})(x::AbstractArray, ps, st) = m((x, x), ps, st)

function (m::ChannelsLast{<:GatedLinearUnit{False}})((xv, xg)::NTuple{2,AbstractArray}, ps, st)
    glu = m.layer
    v, _ = ChannelsLast(glu.linear)(xv, ps.linear, st.linear)
    g, _ = ChannelsLast(glu.gate)(xg, ps.gate, st.gate)
    return v .* glu.activation.(g), st
end

(::ChannelsLast{<:GatedLinearUnit{True}})(::Tuple, ps, st) =
    throw(ArgumentError("Fused GatedLinearUnit does not support Tuple inputs."))
