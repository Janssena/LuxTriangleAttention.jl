module LuxTriangleAttentionReactantExt

using LuxTriangleAttention: LuxTriangleAttention, Attention
using Reactant: AnyTracedRArray

# Reactant traces every element, so a boolean mask under `Reactant.@compile`
# arrives as `TracedRArray{Bool,N} <: AbstractArray{TracedRNumber{Bool},N}`,
# not `AbstractArray{Bool,N}` — plain `mask::AbstractArray{Bool}` dispatch in
# the base package never matches it. `Reactant.AnyTracedRArray{T,N}` is the
# alias Reactant provides for exactly this ("anything holding traced `T`s",
# covering `TracedRArray` itself and reshapes/views of it); give traced Bool
# masks the same codepaths as plain `Bool` masks.

function LuxTriangleAttention._apply_tri_mask!(a::AbstractArray{T}, b, mask::AnyTracedRArray{Bool}) where T
    mask_reshaped = reshape(mask, 1, size(mask)...)
    @. a = ifelse(mask_reshaped, a, zero(T))
    @. b = ifelse(mask_reshaped, b, zero(T))
    return nothing
end

LuxTriangleAttention.ending_permute(mask::AnyTracedRArray{Bool,3}) = permutedims(mask, (2, 1, 3))

(l::Attention)(x::Union{<:Tuple,AbstractArray{T}}, mask::AnyTracedRArray{Bool}, ps, st) where T =
    l(x, nothing, mask, ps, st)

end # module
