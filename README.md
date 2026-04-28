# LuxTriangleAttention.jl

Efficient Triangle Attention and Triangle Multiplication layers for Lux.jl, specifically designed for protein structure prediction models like AlphaFold and Boltz.

## Features

- **Triangle Attention**: Implementation of Starting and Ending Axial Attention with Triangle Bias (AlphaFold: Algorithms 13 & 14).
- **Triangle Multiplication**: Implementation of Outgoing and Incoming Triangle Multiplication (AlphaFold: Algorithms 11 & 12).
- **Lux Native**: Built on top of `Lux.jl` for modularity and performance.
- **Performance Optimized**: Uses `Lux.scaled_dot_product_attention`, `Lux.batched_matmul`, and fused kernels where possible.
- **Flexible Gating**: Support for gated attention and Gated Linear Units (GLU/SwiGLU).

## Numerical Parity

This package has undergone extensive numerical parity testing against reference Python implementations (AlphaFold2, AlphaFold3, and Boltz2). All layers are verified to produce consistent results with their Python counterparts across various precisions and input configurations.


## Installation

```julia
using Pkg
Pkg.add("LuxTriangleAttention")
```

## Quick Start

### Triangle Attention

```julia
using Lux, LuxTriangleAttention, Random

# [Channels, Ni, Nj, Batch]
x = randn(Float32, 64, 32, 32, 1)

# Initialize Layer
model = TriangleAttention(
    64,      # input channels
    32,      # head dimension
    4;       # number of heads
    is_starting=true # true for Triangle Attention Starting Node
)

# Setup Lux
rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

# Forward pass
y, st = model(x, ps, st)
```

### Triangle Multiplication

```julia
model = TriangleMultiplication(
    64,      # input channels
    128,     # hidden channels
    is_outgoing=true # true for Triangle Multiplication Outgoing
)

y, st = model(x, ps, st)
```

## Layer Details

### Triangle Attention
Triangle Attention is a variant of axial attention that incorporates a "triangle bias" derived from the pair representation. This implementation supports:
- **Starting Node Attention**: Attends over the $j$-dimension.
- **Ending Node Attention**: Attends over the $i$-dimension.

**Dimensional Conventions**:
- Input shape: `[C, Ni, Nj, B]`.
- Query shape: `[D, H, Ni, Nj, B]`.
- Attention Dimension: This implementation performs attention over the **3rd dimension** in [D, H, Ni, Nj, B] (`token_dim=3`). This differs from some Python implementations (e.g., Boltz-1) which attend over the 4th dimension.

### Triangle Multiplication
Triangle Multiplication updates the pair representation by combining information from edges sharing a common node.
- **Outgoing**: $\mathbf{z}_{ij} = \sum_k \mathbf{a}_{ik} \odot \mathbf{b}_{jk}$
- **Incoming**: $\mathbf{z}_{ij} = \sum_k \mathbf{a}_{ki} \odot \mathbf{b}_{kj}$

## Masking and Bias

### Boolean Masking
Optional masks of shape `[N, B]`, `[Ni, Nj, B]`, or `[N, S, B]` are automatically reshaped to fit the internal attention score format.

```julia
# Example: Binary mask for sequence padding [N, B]
mask = rand(Bool, 32, 1) 
y, st = model((; x, mask), ps, st)

# Example: Pairwise mask [Ni, Nj, B]
mask_ij = rand(Bool, 32, 32, 1)
y, st = model((; x, mask=mask_ij), ps, st)
```

### Triangle Bias
If using custom attention bias, it must be provided in the correct shape. The `prep_triangle_bias` helper is available to shape bias tensors correctly.

```julia
# 1. Generate raw bias [Heads, Ni, Nj, Batch]
raw_bias = randn(Float32, 4, 32, 32, 1)

# 2. Shape for internal Attention mechanism [Ni, Nj, Heads, 1, Batch]
bias = prep_triangle_bias(raw_bias)

# 3. Pass to model via NamedTuple
y, st = model((; x, bias), ps, st)
```

## License
GNU General Public License