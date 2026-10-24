# 4. Half precision

```@meta
CurrentModule = LuxTriangleAttention
```

Float16 tops out at 65504, and real protein activations get there. The layers handle `Float16` and `BFloat16`
(BFloat16s.jl's type, the one Metal.jl and Reactant use) so that the result stays finite and close to a Float64
reference computed from the same rounded inputs. Casting a model is the usual Lux call:

```julia
using Lux, LuxTriangleAttention, Random

attn = TriangleAttention(32, 8, 2)
ps, st = Lux.setup(Random.default_rng(), attn)
ps16, st16 = Lux.f16(ps), Lux.f16(st)

x16 = Float16.(randn(Float32, 32, 64, 64, 1))
y16, _ = attn(x16, ps16, st16)
```

Two reductions over long axes overflow on real data, and the package treats each:

| where | why it overflows | what is done |
|:--|:--|:--|
| the variance in a LayerNorm | Σx² over a few hundred channels passes 65504 at \|x\| ≈ 20 | [`StableLayerNorm`](@ref) accumulates in Float32 |
| the contraction in triangle multiplication | Σₖ aₖ·bₖ over N positions (``\lvert X\rvert`` ≈ 4.8e5 at N = 494 in AlphaFold2's multimer template stack) | a [`ContractionMode`](@ref), below |
| the attention logits before the `1/√d` scale | 9e4 on Boltz2 activations, if the compiler reassociates the scale | the chunked and kernel paths run in Float32 internally |

## LayerNorm

[`StableLayerNorm`](@ref) is a `Lux.LayerNorm` with Float32 statistics for 16-bit data and *exactly* the wrapped
layer for everything else (bitwise, so Lux's and LuxLib's specialisations for the array type still apply). Its
parameters are the wrapped layer's own, so weights load unchanged. [`TriangleAttention`](@ref) and
[`TriangleMultiplication`](@ref) already use it; use it in your own blocks the same way:

```julia
ln = StableLayerNorm((128,); dims=1)         # same parameter tree as Lux.LayerNorm((128,); dims=1)
ln0 = LayerNormNoBias((128,); dims=1)        # bias frozen at zero
```

## Triangle multiplication: three contraction modes

The contraction picks one path, statically, from the data's element type and the layer's state:

| mode | when | what runs |
|:--|:--|:--|
| [`FullPrecision`](@ref) | not 16-bit data | the ordinary code path |
| [`HalfSafe`](@ref) | 16-bit data, ordinary parameters (e.g. a plain `Lux.f16(ps)`) | the contraction in Float32: always correct, slower and wider |
| [`Folded`](@ref)`(k)` | the projections were pre-scaled by 2⁻ᵏ | the contraction sums roughly a mean, so it stays in range in Float16; fast and exact |

Without doing anything, 16-bit data takes `HalfSafe`. To get `Folded`, scale the parameters once with
[`fold_trimul`](@ref), before casting to 16 bits, for the longest sequence they must serve ([`fold_k`](@ref)):

```julia
using LuxTriangleAttention: fold_trimul, fold_k    # public, not exported

mul = TriangleMultiplication(128, 128)
ps, st = Lux.setup(Random.default_rng(), mul)

ps, st = fold_trimul(ps, st, fold_k(1024))   # 4ᵏ ≥ 1024  →  k = 5
ps16, st16 = Lux.f16(ps), Lux.f16(st)

y16, _ = mul(x16, ps16, st16)
```

`fold_k(L)` is the smallest `k` with `4ᵏ ≥ L`. Folding multiplies each of the two value projections by 2⁻ᵏ, so the
contraction is divided by 4ᵏ, and the output LayerNorm (scale-invariant, with its ε scaled by 16⁻ᵏ) removes it
exactly. A larger `k` than needed only makes values smaller, which Float16 represents at full relative precision down
to 6e-5, so `L` is a bound on the sequence length, not the length of any particular input.

The mode is decided by the *state*: `st.core.fold_k` exists exactly when the weights carry the 2⁻ᵏ, whatever the
element type. A layer can therefore never apply a 4ᵏ its weights do not carry, nor miss one they do. An `Int` in a Lux
state is a compile-time constant under Reactant, so the choice is free at runtime. `contraction_mode(x, st.core)`
(see [`contraction_mode`](@ref)) shows what a call will use.

## Attention

- **Masks** are applied by selecting [`mask_value`](@ref)`(T)` (`-floatmax(T)/2`, finite in every precision), never
  by adding `-Inf` or `1e9·(mask - 1)`. The latter is `-Inf` in Float16 (`Float16(-1e9) == -Inf`), which turned a fully
  padded window into `NaN`. A row with no live key attends uniformly everywhere (see [`SDPA`](@ref)).
- **Chunked attention** is Float16-safe on the CPU and, rolled, under Reactant (a chunk runs in Float32 there). The
  plain `use_kernel=false, chunk_size=nothing` path is LuxLib's attention, which is *not* Float16-safe under Reactant
  and materialises every score; use `chunk_size` or `use_kernel` for 16-bit data in production.
- **Fused kernels** accumulate in Float32 whatever the storage type. A bias in another precision is converted with
  saturation at `±floatmax(T)/2` rather than overflowing to `±Inf`.

!!! warning "Peaked attention in the backward pass"
    The kernels take Float16 operands for their matrix products, the FlashAttention-2 convention. A loss-scaled `dO`
    large enough to push `dS = P·(dP − δ)` past 65504 gives non-finite gradients, as it would there. Whether that
    happens depends on the inputs (peaked attention together with a large aligned `|v|·|dO|`), not on the
    configuration, and it is not detectable by the tuner. On random problems `dS` stays far from overflow even at
    `dO` ≈ 2e4.

## Precision utilities

These are public but not exported (`LuxTriangleAttention.ishalf`, ...). [`ishalf`](@ref)`(T)` says whether `T` is a 16-bit float type, [`mask_value`](@ref) gives the finite masking value for a
type or array, and [`layernorm_f32stats`](@ref) is the one-pass Float32-statistics normalisation behind
`StableLayerNorm`, for use in your own layers.
