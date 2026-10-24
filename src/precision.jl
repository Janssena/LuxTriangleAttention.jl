# Float16 tops out at 65504. Two kinds of reduction over long axes overflow on real activations:
# the sum of squares behind a LayerNorm's variance, and a contraction Σₖ aₖ·bₖ over hundreds of
# positions (triangle multiplication, BioFoldCore's outer product mean). A layer with such a
# contraction picks one of three paths:
#
#   Folded(k)        the projections feeding the contraction were pre-scaled by 2⁻ᵏ (`fold_trimul`),
#                    so it sums ~a mean, not a sum, and stays in range; the layer undoes the 4ᵏ (or
#                    normalises it away). Exact in ANY precision — it is what makes Float16 fast.
#   HalfSafe()       16-bit data, unscaled parameters (e.g. a plain `Lux.f16(ps)`): the contraction
#                    runs in Float32. Always correct, just slower and wider.
#   FullPrecision()  otherwise: the ordinary code path, unchanged.
#
# `Folded` is chosen by the layer STATE (`st.fold_k`), which only the function that scaled the
# weights writes: `fold_k` in the state ⇔ the weights carry 2⁻ᵏ, whatever the element type, so a layer
# can never apply a 4ᵏ its weights do not carry, nor miss one they do. An `Int` in a Lux state is a
# compile-time constant under Reactant, so the choice costs nothing at runtime.

"""
    ishalf(T::Type) -> Bool

Whether `T` is a 16-bit floating-point type that needs the [`ContractionMode`](@ref) handling:
`Float16`, `BFloat16` (BFloat16s.jl's, which Metal.jl and Reactant use) and `Core.BFloat16`.
"""
ishalf(::Type) = false
ishalf(::Type{Float16}) = true
ishalf(::Type{BFloat16}) = true
# BFloat16s.jl 0.6 on Julia 1.12 IS `Core.BFloat16`; a second method for the same type would overwrite the first
# (an error during precompilation), so it is added only where the two types differ.
@static if isdefined(Core, :BFloat16) && BFloat16 !== Core.BFloat16
    ishalf(::Type{Core.BFloat16}) = true
end

"""
    mask_value(T::Type) -> T
    mask_value(x::AbstractArray) -> eltype

The value to put where attention is masked: `-floatmax(T) / 2`, FINITE in every precision. A masked
logit written as `-Inf` (what `typemin` is for a float, and what LuxLib fills in), or as a large constant
that overflows `T`, turns a row with no live key into `-Inf - (-Inf) = NaN`; OpenFold's and Boltz2's
`1e9 * (mask - 1)` is finite in Float32 but IS `-Inf` in Float16 (`Float16(-1e9) == -Inf`), which is how a
fully padded window gave NaN there. Use it by SELECTING, not adding:

    logits = ifelse.(keep, logits, mask_value(T))

A select needs only that the value is finite, so it holds in any precision, and a row where every logit is
`mask_value` is a uniform softmax (the behaviour of a fully masked row everywhere in this package, see
[`SDPA`](@ref)). As an additive bias it still fits `T` for logits of magnitude below `floatmax(T) / 2`
(about 3.3e4 in Float16); a select is the safe form. `exp(mask_value(T) - m)` is exactly `0` for any
row maximum `m` a model can produce, so masked keys get no weight in a row that has a live one.

BFloat16 (`Core.BFloat16`, and Reactant's) has Float32's exponent range and gets Float32's value.
"""
mask_value(::Type{T}) where {T<:AbstractFloat} = -floatmax(T) / 2
function mask_value(::Type{T}) where {T}
    R = realtype(T)
    return R === T ? T(mask_value(Float32)) : mask_value(R) # TODO: maybe make this explicit
end
mask_value(x::AbstractArray) = mask_value(realtype(eltype(x)))

"""
    fold_k(L::Integer) -> Int

The scale exponent for a contraction of length up to `L`: the smallest `k` with `4ᵏ ≥ L`. Folding
2⁻ᵏ into each of the two projections divides the contraction by 4ᵏ, so it sums roughly a mean whose
size no longer grows with `L`. A `k` larger than needed only leaves values smaller, which Float16
represents at full relative precision down to 6e-5, so `L` is a BOUND (the longest contraction the
parameters must serve), not the length of any particular input.
"""
fold_k(L::Integer) = ceil(Int, log2(max(L, 1)) / 2)

"""
    ContractionMode

How a layer with a long contraction runs, chosen from its input's element type and its state:
[`Folded`](@ref), [`HalfSafe`](@ref) or [`FullPrecision`](@ref) (see `src/precision.jl`).
"""
abstract type ContractionMode end
"""Parameters not pre-scaled, data not 16-bit: the ordinary code path."""
struct FullPrecision <: ContractionMode end
"""16-bit data, parameters NOT pre-scaled: the contraction runs in Float32."""
struct HalfSafe <: ContractionMode end
"""The contraction's projections pre-scaled by 2⁻ᵏ (`st.fold_k == k`), any element type."""
struct Folded <: ContractionMode
    k::Int
end

"""
    contraction_mode(x::AbstractArray, st) -> ContractionMode

[`Folded`](@ref)`(st.fold_k)` if the state carries one; else [`HalfSafe`](@ref) if `x` holds 16-bit
numbers (traced or not); else [`FullPrecision`](@ref).
"""
contraction_mode(x::AbstractArray, st) =
    _has_fold(st) ? Folded(st.fold_k) : ishalf(realtype(eltype(x))) ? HalfSafe() : FullPrecision()
_has_fold(st::NamedTuple) = haskey(st, :fold_k)
_has_fold(_) = false
