# LuxTriangleAttention.jl

```@meta
CurrentModule = LuxTriangleAttention
```

Triangle Attention and Triangle Multiplication layers for [Lux.jl](https://github.com/LuxDL/Lux.jl), built for
protein structure prediction models such as AlphaFold and Boltz.

- [`TriangleAttention`](@ref): starting and ending node attention with a triangle bias (AlphaFold2 Algorithms 13 and 14).
- [`TriangleMultiplication`](@ref): outgoing and incoming triangle multiplication (AlphaFold2 Algorithms 11 and 12).
- [`Attention`](@ref): the multi-head attention both are built on, usable on its own (self- and cross-attention,
  gating, pair bias, masks).
- Fused attention kernels for Apple GPUs, tuned for your device, that run eagerly and inside a
  [Reactant](https://github.com/EnzymeAD/Reactant.jl)-compiled program.
- Float16 and BFloat16 that stay finite on real activations.
- Numerically verified against the PyTorch references of AlphaFold2, AlphaFold3 and Boltz2.

## Installation

```julia
using Pkg
Pkg.add("LuxTriangleAttention")
```

Julia 1.12 is required. The GPU kernels need [Metal.jl](https://github.com/JuliaGPU/Metal.jl) (load it with
`using Metal`), and compiled programs need [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl); both are
optional and enable package extensions.

## Quick start

Inputs are channels-first: `[C, Ni, Nj, B]`.

```julia
using Lux, LuxTriangleAttention, Random

x = randn(Float32, 64, 32, 32, 1)

attn = TriangleAttention(64, 32, 4; is_starting=true)     # channels, head dimension, heads
ps, st = Lux.setup(Random.default_rng(), attn)
y, st = attn(x, ps, st)                                   # [64, 32, 32, 1]

mul = TriangleMultiplication(64, 128; is_outgoing=true)   # channels, hidden channels
ps, st = Lux.setup(Random.default_rng(), mul)
y, st = mul(x, ps, st)
```

The [first tutorial](tutorials/1-layers.md) covers the layers, masks and the bias layouts in detail.

## Where to go next

| | |
|:--|:--|
| [Layers, masks and bias](tutorials/1-layers.md) | the layers, their shapes, boolean masks, `prep_bias` and the `:qk` / `:kq` bias layouts |
| [Fused attention kernels](tutorials/2-fused-kernels.md) | `use_kernel`, the two built-in kernels, tuning and persisting the result |
| [Compiling with Reactant](tutorials/3-reactant.md) | `@compile`, `chunk_size` and why the chunk loop must stay rolled |
| [Half precision](tutorials/4-half-precision.md) | Float16 and BFloat16: what overflows, and the three contraction modes |
| [Your own attention kernel](tutorials/5-custom-kernels.md) | registering a `.metallib` or `.cubin` as an attention kernel |
| [Kernel operands](kernel-operands.md) | the shapes, element types and argument lists every attention kernel reads |
| [API reference](api.md) | every exported and public name |

## Dimension conventions

Attention here runs over the **third** dimension (`token_dim=3`) of a `[C, Ni, Nj, B]` input, with the queries
`[head_dim, H, Ni, Nj, B]`. Many Python implementations (AlphaFold, Boltz) attend over the fourth dimension of a
`[B, S, N, C]` tensor, so a Python bias or mask needs the permutation that [`prep_bias`](@ref) and
[`prep_mask`](@ref) apply. The two [bias layouts](tutorials/1-layers.md#Bias-layouts) exist for exactly this.

## Kernels are KernelDispatch ops

The kernel machinery (selection, variants, launching binaries, embedding a kernel in a Reactant program, the Enzyme
rule) is [KernelDispatch.jl](https://github.com/Janssena/KernelDispatch.jl). Attention is one op there, [`SDPA`](@ref),
and the built-in kernels are two of its kernels. Bringing your own is the
[last tutorial](tutorials/5-custom-kernels.md); the KernelDispatch documentation explains the concepts it builds on.

## Backends

- **Layers**: ordinary Julia on any array type Lux supports, on the host or a GPU, and compiled with Reactant
  (CPU, or an Apple GPU through the jax-mps PJRT plugin).
- **Fused kernels**: `:default` and `:flash_mma` run on Metal. On the host, `use_kernel=true` runs the ordinary
  softmax attention, so a model built for the GPU still runs on the CPU. On any other device the kernel has to be
  [your own](tutorials/5-custom-kernels.md) (a `.cubin`, say); KernelDispatch supports CUDA, but this package does
  not build kernels for it.
