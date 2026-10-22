module LuxTriangleAttentionReactantExt

# What is specific to attention under Reactant: traced boolean masks reaching the layers' mask
# handling, and the rolled `@trace` chunking path. Running kernels from a trace — `backend_of` for
# traced arrays, custom-call embedding, the traced reverse — is KernelDispatch' Reactant extension.

using LuxTriangleAttention: LuxTriangleAttention, Attention, StableLayerNorm, BFloat16
using Reactant: Reactant, AnyTracedRArray, TracedRNumber
using Reactant.ReactantCore: @trace

# Reactant traces every element, so a boolean mask under `Reactant.@compile`
# arrives as `TracedRArray{Bool,N} <: AbstractArray{TracedRNumber{Bool},N}`,
# not `AbstractArray{Bool,N}` — plain `mask::AbstractArray{Bool}` dispatch in
# the base package never matches it. `Reactant.AnyTracedRArray{T,N}` is the
# alias Reactant provides for exactly this ("anything holding traced `T`s",
# covering `TracedRArray` itself and reshapes/views of it); give traced Bool
# masks the same codepaths as plain `Bool` masks.
LuxTriangleAttention.ending_permute(mask::AnyTracedRArray{Bool,3}) =
    permutedims(mask, (2, 1, 3))

(l::Attention)(x::Union{<:Tuple,AnyTracedRArray{T}}, mask::AnyTracedRArray{Bool}, ps, st) where T =
    l(x, nothing, mask, ps, st)

# A traced 16-bit array is `AbstractArray{TracedRNumber{Float16}}`, so the base package's
# `AbstractArray{T<:Union{Float16,BFloat16}}` method for `StableLayerNorm` never matches it and the
# wrapped `Lux.LayerNorm` would accumulate in 16 bits. Route it to the Float32-statistics path.
(m::StableLayerNorm)(x::AnyTracedRArray{T}, ps, st::NamedTuple) where {T<:Union{Float16,BFloat16}} =
    LuxTriangleAttention._layernorm_f32stats(m.layer, x, ps, st), st

# The rolled `@trace` chunking path (`_sdpa_enzyme_safe`, `chunked_compile_options`, `check_rolled`,
# `_chunked_sdpa` and its helpers) lives in `ext/reactant/chunk.jl`.
include("LuxTriangleAttentionReactantExt/chunk.jl")

end # module LuxTriangleAttentionReactantExt
