# Triangle Attention: A Comparative Analysis

> Detailed comparison of Triangle Attention implementations in Boltz2, OpenFold1, and OpenFold3. Mathematical formulation, algorithmic differences, and Julia implementation guide.

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Algorithm Anatomy](#3-algorithm-anatomy)
4. [Software Architecture: Static Dispatch](#4-software-architecture-static-dispatch)
5. [Tensor Layouts & Dimensions](#5-tensor-layouts--dimensions)
6. [Comparative Analysis](#6-comparative-analysis)
7. [Implementation Divergence](#7-implementation-divergence)
8. [Reference](#8-reference)

---

## 1. Overview

### What is Triangle Attention?

Triangle Attention is a specialised attention mechanism used in protein structure prediction models. Unlike standard attention that processes sequences, triangle attention processes the **pair representation** — a 2D matrix representing relationships between all pairs of amino acid residues.

The pair representation is a matrix of shape $[N, N, C]$ where:
- $N$ = number of residues (e.g., 128-1024 for proteins)
- $C$ = number of features per position pair
- Each cell $(i, j)$ contains a feature vector representing the relationship between residues $i$ and $j$

```
Pair Representation [N × N × C]:

        residue j
      ┌─────┬─────┬─────┬─────┐
      │     │     │     │     │
    i ├─────┼─────┼─────┼─────┤  Each cell (i,j) contains
      │     │     │     │     │  a feature vector of dim C
      ├─────┼─────┼─────┼─────┤  representing the relationship
      │     │     │     │     │  between residues i and j
      ├─────┼─────┼─────┼─────┤
      │     │     │     │     │
      └─────┴─────┴─────┴─────┘
```

### Implementations Compared

| Implementation | File | Kernel Options |
|---------------|------|----------------|
| **OpenFold1** | `openfold/model/triangular_attention.py` | Memory-efficient, DeepSpeed, CueEquiv, LMA |
| **OpenFold3** | `openfold3/core/model/layers/triangular_attention.py` | DeepSpeed, CueEquiv, LMA |
| **Boltz2** | `boltz/model/layers/triangular_attention/attention.py` | use_kernels (CueEquiv/Trifast) |

---

## 2. Mathematical Foundations

### Standard Attention (Reference)

Standard Self-Attention computes an output matrix $O$ from Query ($Q$), Key ($K$), and Value ($V$) matrices:

$$O = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

where $Q, K, V \in \mathbb{R}^{N \times d}$.

### Triangle Attention

Triangle attention extends this to the **pair representation** $\mathbf{z} \in \mathbb{R}^{N \times N \times C}$. For each pair $(i, j)$, the mechanism attends to a set of related pairs $(i, k)$ or $(k, j)$ to update the relationship between $i$ and $j$.

#### Starting Node Variant (Algorithm 13)

For each pair $(i, j)$, we aggregate information from all positions $k$ along the **same row** $i$:

$$
\begin{aligned}
S_{ijk} &= \frac{Q_{ij} \cdot K_{ik}}{\sqrt{d}} + B_{jk} \\
A_{ijk} &= \text{softmax}_k(S_{ijk}) \\
O_{ij} &= \sum_k A_{ijk} V_{ik}
\end{aligned}
$$

**Core Logic:**
- **Fixed Query Position:** Row $i$ is constant during the reduction.
- **Normalization:** The softmax is computed over the $k$ index (attending TO all $k$ in row $i$).
- **Context:** Tells position $(i, j)$ how it relates to everything else on row $i$.

#### Ending Node Variant (Algorithm 14)

For each pair $(i, j)$, we aggregate information from all positions $k$ along the **same column** $j$:

$$
\begin{aligned}
S_{ijk} &= \frac{Q_{ij} \cdot K_{kj}}{\sqrt{d}} + B_{ik} \\
A_{ijk} &= \text{softmax}_k(S_{ijk}) \\
O_{ij} &= \sum_k A_{ijk} V_{kj}
\end{aligned}
$$

**Core Logic:**
- **Fixed Query Position:** Column $j$ is constant during the reduction.
- **Normalization:** The softmax is computed over the $k$ index (attending FROM all $k$ in column $j$).
- **Context:** Tells position $(i, j)$ how it relates to everything else in column $j$.

#### Visualizing Information Flow

```text
STARTING NODE (Row-wise)             ENDING NODE (Column-wise)
Aggregates along row i               Aggregates along column j

      k=1 k=2 k=3 k=N                      j
    ┌───┬───┬───┬───┐                ┌───┐ k=1
    │   │   │   │   │                │   │
 i ─┤ Q ─► K ─► K ─► K               │ K │ k=2
    │   │   │   │   │                │   │
    └───┴───┴───┴───┘                │ K │ k=3
          j                          │   │
                                     │ ▼ │
    Attend TO others                 │ Q │ i
    on the same row                  │   │
                                     └───┘
                               Attend FROM others
                               on the same column
```

---

## 3. Algorithm Description

### Algorithm 12: Generic Triangle Attention Layer

All three implementations follow Algorithm 12 as the base layer structure:

```
┌────────────────────────────────────────────────────────────────┐
│                    ALGORITHM 12: TRIANGLE ATTENTION LAYER      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:  x [*, N, N, c_in]     pair representation           │
│          mask [*, N, N]         optional attention mask        │
│                                                                 │
│  1. LayerNorm:                                                │
│     x_norm = LayerNorm(x)                                      │
│                                                                 │
│  2. Triangle Bias (learnable):                                │
│     bias = Linear(x_norm)  →  [*, N, N, num_heads]            │
│     bias = permute(bias, (3, 1, 2))  →  [num_heads, *, N, N] │
│                                                                 │
│  3. Q, K, V Projections:                                       │
│     Q = x_norm @ W_q                                           │
│     K = x_norm @ W_k                                           │
│     V = x_norm @ W_v                                           │
│                                                                 │
│  4. Apply Attention:                                           │
│     if starting:                                               │
│         O = TriangleAttentionStarting(Q, K, V, bias, mask)    │
│     else:                                                      │
│         O = TriangleAttentionEnding(Q, K, V, bias, mask)       │
│                                                                 │
│  5. Output Projection:                                          │
│     O = O @ W_o                                                │
│                                                                 │
│  Output: x [*, N, N, c_in]                                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Attention Computation (Starting Node)

```
┌────────────────────────────────────────────────────────────────┐
│                 STARTING NODE ATTENTION                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each (i, j, h):                                          │
│                                                                 │
│  1. Compute attention scores:                                 │
│     s[j,k] = (Q[i,j,h] · K[i,k,h]) / √d  +  bias[h,j,k]     │
│                                                                 │
│  2. Apply mask (set masked positions to -∞):                   │
│     if mask[i,k] == 0:  s[j,k] = -∞                          │
│                                                                 │
│  3. Softmax over k:                                           │
│     p[j,k] = exp(s[j,k] - max_k(s[j,*]))                     │
│     p[j,k] = p[j,k] / Σ_k p[j,k]                             │
│                                                                 │
│  4. Weighted sum:                                             │
│     O[i,j,h] = Σ_k p[j,k] · V[i,k,h]                         │
│                                                                 │
│  Key: Softmax is over dimension k (row dimension)             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Ending Node Transposition

The ending node variant is implemented by transposing the input before and after attention:

```python
# For ending node:
x = x.transpose(-2, -3)    # swap i and j dimensions
mask = mask.transpose(-1, -2)

# Now ending node becomes starting node on transposed data
x = self.mha(q_x=x, kv_x=x, biases=biases, ...)

x = x.transpose(-2, -3)   # transpose back
```

---

## 4. Software Architecture: Static Dispatch

To achieve zero-overhead abstractions, we leverage Julia's **Multiple Dispatch** and `Static.jl`. By storing properties directly in the layer's metadata, we enable specialized compile-time optimizations.

### Configuration via StaticBool

Instead of runtime `if` statements, we store the `starting` property as a `StaticBool` (`True` or `False`) within the layer struct. This allows the compiler to select the appropriate algorithm variant at compile-time.

```julia
using Static: static, True, False

# Layer structure stores StaticBool
struct TriangleAttention{S<:StaticBool, ...} <: Lux.AbstractExplicitLayer
    is_starting::S
    # ...
end

function forward(m::TriangleAttention, x, ps, st)
    # Selection via Multiple Dispatch directly on the struct field
    return _forward_logic(m.is_starting, m, x, ps)
end

# Specialization for Starting Node (Algorithm 13)
function _forward_logic(::True, m, x, ps)
    # Row-wise optimization: no/specific transpose required
    return _triangle_attention_kernel(x, ...)
end

# Specialization for Ending Node (Algorithm 14)
function _forward_logic(::False, m, x, ps)
    # Col-wise optimization: transpose once, perform row-wise attn
    x_t = PermutedDimsArray(x, (2, 1, 3, 4))
    return _triangle_attention_kernel(x_t, ...)
end
```

### Backend-Specific Kernels

The high-level logic dispatches to optimized backend kernels based on the array type (e.g., `CuArray` vs. `AbstractArray`).

```julia
# Dispatches to CPU-optimized LoopVectorization kernel
_triangle_attention_kernel(q::AbstractArray, ...) = 
    _triangle_attention_kernel_cpu(q, ...)

# Dispatches to specialized CUDA kernel
_triangle_attention_kernel(q::CuArray, ...) = 
    _triangle_attention_kernel_cuda(q, ...)
```

---

## 5. Tensor Layouts & Dimensions

### Dimension Reference

| Symbol | Meaning | Typical Values |
|--------|---------|---------------|
| `B` | Batch size (number of protein sequences) | 1-64 |
| `N` | Sequence length (number of residues per protein) | 64-4096 |
| `H` | Number of attention heads | 4-64 |
| `D` | Feature dimension per head | 32-128 |
| `C_in` | Input channel dimension (total features) | 128-1024 |

**Note:** `D × H = C_in` (features are split across heads)

### Input/Output Dimensions

| Tensor | Python Shape | Julia Shape (before MHA) | Julia Shape (after MHA) | Description |
|--------|--------------|-------------------------|------------------------|-------------|
| Input `x` | `[B, N, N, C_in]` | `[C_in, N, N, B]` | — | Pair representation: for each protein in batch, an $N \times N$ matrix where cell $(i, j)$ contains a $C_in$-dimensional feature vector |
| Q, K, V (pre-split) | `[B, N, N, C_in]` | `[C_in, N, N, B]` | — | Before multi-head attention reshape |
| Q, K, V (per-head) | `[B, N, N, H, D]` | — | `[D, H, N, N, B]` | After reshape: `D × H = C_in` |
| `mask` | `[B, N, N]` | `[N, N, B]` | `[N, N, B]` | Attention mask: for each protein, a binary $N \times N$ matrix where `mask[i,k] = 1` means position $k$ is valid for attention |
| Triangle bias | `[B, H, N, N]` | `[H, N, N, B]` | `[H, N, N, B]` | Learnable bias per head and batch element |
| Output | `[B, N, N, C_in]` | `[C_in, N, N, B]` | — | Same shape as input |

**Note:** The attention computation operates on `[D, H, N, N, B]` shaped tensors (per-head format). Transform via `reshape(x, D, H, N, N, B)` where `D × H = C_in`.

**Key dimension differences between Python and Julia:**

```
Python:  [B, N, N, C_in]    # Row-major: batch outermost, features innermost
Julia:   [C_in, N, N, B]    # Column-major: features outermost, batch innermost

Shape breakdown (Python):     Shape breakdown (Julia):
[B, N, N, C_in]              [C_in, N, N, B]
│  │  │    │                  │     │  │  │
│  │  │    └── C_in features  │     │  │  └── B proteins
│  │  └─── N columns (j)      │     │  └─── N columns (j)
│  └────── N rows (i)         │     └────── N rows (i)
└──────── B proteins          └────────── C_in features
```

**Visualizing Input Shape `[B, N, N, C_in]`:**

```
For batch element b = 1:
    
    Pair representation x[b]:  [N × N × C_in]
    
        residue j →
    i  ┌─────────────────────────────────────┐
    ↓  │  x[b,i,j,:] ──► C_in features      │
       │     │                                │
       │     ▼                                │
       │  This is a single "row" of N cells, │
       │  each with C_in features             │
       └─────────────────────────────────────┘
    
    Shape breakdown:  [B, N, N, C_in]
                       │  │  │    │
                       │  │  │    └─── C_in features per cell
                       │  │  └──────── N columns (residue j)
                       │  └─────────── N rows (residue i)
                       └──────────────── B proteins in batch
```

### Mask Bias Dimensions

The mask bias converts a binary mask into an additive bias for attention:

```
mask [B, N, N]  ──►  mask_bias [broadcasted to match Q shape]

mask[i,k] = 1  →  add 0 to attention score (position is valid)
mask[i,k] = 0  →  add -∞ to attention score (position is masked)
```

**Formula:** `mask_bias = inf * (mask - 1)`

| Implementation | Broadcasting | Resulting Shape |
|----------------|--------------|-----------------|
| OpenFold1/3 | `[..., :, None, None, :]` | `[B, N, 1, 1, N]` |
| Boltz2 | `[..., None, None, :, :]` | `[B, 1, 1, N, N]` |

See [Section 6](#mask-bias-dimensions) for detailed explanation of why these differ.

### Internal Attention Dimensions (per head)

After Q, K, V are reshaped for multi-head attention:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `q` | `[B, N, N, H, D]` | Query per head (H heads × D features each) |
| `k`, `v` | `[B, N, N, H, D]` | Key and Value per head |
| Attention scores | `[B, H, N, N, N]` | Scores for each query-position × key-position pair |

**Note:** The attention score shape `[B, H, N, N, N]` means:
- For each batch element `b`
- For each head `h`
- For each query position `i` and output column `j`
- Compute attention weights over all key positions `k`

### Layout Conversion

```julia
# Julia [C_in, N, N, B] → Python [B, N, N, C_in]
julia_to_python(q) = permutedims(q, (4, 2, 3, 1))

# Python [B, N, N, C_in] → Julia [C_in, N, N, B]
python_to_julia(q) = permutedims(q, (4, 2, 3, 1))
```

---

## 6. Comparative Analysis

### OpenFold1

**File:** `python/openfold/openfold/model/triangular_attention.py`

```python
class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf
        
        self.layer_norm = LayerNorm(self.c_in)
        self.linear = Linear(c_in, self.no_heads, bias=False, init="normal")
        self.mha = Attention(c_in, c_in, c_in, c_hidden, no_heads)
    
    def forward(self, x, mask=None, chunk_size=None,
                use_memory_efficient_kernel=False,
                use_deepspeed_evo_attention=False,
                use_cuequivariance_attention=False,
                use_lma=False,
                inplace_safe=False):
        if mask is None:
            mask = x.new_ones(x.shape[:-1])  # [*, N, N]
        
        # Transpose for ending node
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)
        
        x = self.layer_norm(x)
        
        # Mask bias: [*, N, N] → [*, N, 1, 1, N]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        
        # Triangle bias: [*, N, N, num_heads] → [1, num_heads, 1, *, N]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)
        
        biases = [mask_bias, triangle_bias]
        
        if chunk_size is not None:
            x = self._chunk(x, biases, chunk_size, ...)
        else:
            x = self.mha(
                q_x=x, kv_x=x, biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
            )
        
        if not self.starting:
            x = x.transpose(-2, -3)
        
        return x
```

**Key Features:**
- Multiple kernel optimization options
- `use_memory_efficient_kernel`: Memory-efficient attention
- `use_deepspeed_evo_attention`: DeepSpeed Evoformer attention
- `use_cuequivariance_attention`: CueEquivariance kernels
- `use_lma`: Linear multi-head attention

---

### OpenFold3

**File:** `python/openfold-3/openfold3/core/model/layers/triangular_attention.py`

```python
class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting=True, 
                 inf=1e9, linear_init_params=lin_init.tri_att_init):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf
        
        self.layer_norm = LayerNorm(self.c_in)
        self.linear_z = Linear(c_in, self.no_heads, **linear_init_params.linear_z)
        self.mha = Attention(
            c_q=self.c_in, c_k=self.c_in, c_v=self.c_in,
            c_hidden=self.c_hidden, no_heads=self.no_heads,
            linear_init_params=linear_init_params.mha,
        )
    
    def forward(self, x, mask=None, chunk_size=None,
                use_deepspeed_evo_attention=False,
                use_cueq_triangle_kernels=False,
                use_lma=False,
                inplace_safe=False):
        # Same structure as OpenFold1 with:
        # - linear_z instead of linear
        # - use_cueq_triangle_kernels instead of use_cuequivariance_attention
        # - Custom linear initialization
        ...
```

**Key Differences from OpenFold1:**
- Uses `linear_z` instead of `linear` (naming consistency)
- Uses `use_cueq_triangle_kernels` instead of `use_cuequivariance_attention`
- No `use_memory_efficient_kernel` option
- Custom linear initialization (`linear_init_params.tri_att_init`)

---

### Boltz2

**File:** `python/boltz/src/boltz/model/layers/triangular_attention/attention.py`

```python
class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf
        
        self.layer_norm = LayerNorm(self.c_in)
        self.linear = Linear(c_in, self.no_heads, bias=False, init="normal")
        self.mha = Attention(c_in, c_in, c_in, c_hidden, no_heads)
    
    def forward(self, x, mask=None, chunk_size=None, use_kernels=False):
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)
        
        x = self.layer_norm(x)
        
        # Triangle bias: same as OpenFold
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)
        
        # Mask bias: DIFFERENT broadcasting pattern!
        # [*, N, N] → [*, 1, 1, N, N]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        
        biases = [mask_bias, triangle_bias]
        
        # Chunking disabled when using kernels
        if chunk_size is not None and not use_kernels:
            x = self._chunk(x, biases, chunk_size, ...)
        else:
            x = self.mha(q_x=x, kv_x=x, biases=biases, use_kernels=use_kernels)
        
        if not self.starting:
            x = x.transpose(-2, -3)
        
        return x
```

**Supporting: primitives.py**

```python
class Attention(nn.Module):
    def forward(self, q, k, v, bias=None, use_kernels=False, mask=None, scale=None):
        if use_kernels:
            # Use CueEquivariance CUDA kernels or Trifast
            return kernel_triangular_attn(q, k, v, bias, mask=mask, scale=scale)
        else:
            # Standard softmax attention
            return _attention(q, k, v, bias)
```

**Key Differences:**
- Simpler API: just `use_kernels` parameter
- **Different mask bias broadcasting** (see Section 6)
- Chunking disabled when `use_kernels=True`
- Supports Trifast for CPU acceleration

---

## 7. Implementation Divergence

### Summary of Differences

| Feature | OpenFold1 | OpenFold3 | Boltz2 |
|---------|-----------|-----------|--------|
| **Linear layer name** | `self.linear` | `self.linear_z` | `self.linear` |
| **CueEquiv option** | `use_cuequivariance_attention` | `use_cueq_triangle_kernels` | `use_kernels` |
| **Memory-efficient kernel** | Yes | No | No |
| **DeepSpeed Evo** | Yes | Yes | No |
| **LMA option** | Yes | Yes | No |
| **Chunking with kernels** | Yes | Yes | No |
| **Init params** | Default | Custom | Default |
| **Mask bias shape** | `[B, N, 1, 1, N]` | `[B, N, 1, 1, N]` | `[B, N, 1, 1, N]` |

### Mask Bias Dimensions

#### Why Mask Bias Exists

The mask bias converts a binary mask into an additive bias for attention:

```
mask[i,k] = 1  →  attention score unchanged (add 0)
mask[i,k] = 0  →  attention score = -∞ (masked out)

Formula: mask_bias = inf * (mask - 1)

Example with inf=1e9:
  mask = 1  →  mask_bias = 1e9 * (1 - 1) = 0
  mask = 0  →  mask_bias = 1e9 * (0 - 1) = -1e9 ≈ -∞
```

#### Broadcasting for Attention

The attention score has shape $[B, H, N, N, D]$ (simplified). The mask needs to broadcast correctly:

```
Attention scores: [B, H, i, j, d] × [B, H, i, k, d]
                                    ↓
                              [B, H, N, N, N]  (i,j pairs × k positions)
```

The mask bias needs to add to scores where:
- $i$ dimension: matches query position
- $k$ dimension: matches key position

#### OpenFold1/3 Mask Bias

```python
mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
# Shape: [B, N, N] → [B, N, 1, 1, N]
```

```
mask_bias[b, i, 1, 1, k]  adds to  score[b, h, i, j, d]

For fixed i and k, this value is broadcast across all j and h.
```

This works because the attention in OpenFold uses:
- `q[b, i, j, h, d]` — query for position pair (i,j)
- `k[b, i, k, h, d]` — key for position pair (i,k) ← same i!

#### Boltz2 Mask Bias

```python
# Boltz2 mask bias reshaping
mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
# Shape: [B, N, N] → [B, N, 1, 1, N]
```

```text
mask_bias[b, i, 1, 1, k]  adds to  score[b, h, i, j, k]

For fixed i and k, this value is broadcast across all j and h.
```

**Why different?** Boltz uses a different attention formulation where:
- The attention is computed with a different indexing scheme
- The mask applies across all query positions for each key position

#### Detailed Comparison

| Aspect | OpenFold1/3 | Boltz2 |
|--------|-------------|--------|
| **Input mask shape** | `[B, N, N]` | `[B, N, N]` |
| **Broadcasting** | `[..., :, None, None, :]` | `[..., None, None, :, :]` |
| **Resulting mask_bias shape** | `[B, N, 1, 1, N]` | `[B, N, 1, 1, N]` |
| **Broadcasts over** | j dimension | j dimension |

**Why the difference matters:**

In **OpenFold1/3**, the mask broadcasts as:
- `mask_bias[b, i, 1, 1, k]` adds to all `score[b, h, i, j, d]` for all `j`
- This means: for position pair `(i, j)`, mask `[i, k]` controls which `k` positions are valid

In **Boltz2**, the mask broadcasts as:
- `mask_bias[b, 1, 1, i, k]` adds to all `score[b, h, i, j, d]` for all `j`
- This is equivalent mathematically but arranged differently for the kernel

The difference arises because Boltz's kernel implementation requires a different memory layout for efficient CUDA execution.

### Kernel Options Comparison

| Option | OpenFold1 | OpenFold3 | Boltz2 |
|--------|-----------|-----------|--------|
| Memory Efficient | Yes | No | No |
| DeepSpeed Evo | Yes | Yes | No |
| CueEquivariance | Yes | Yes | Yes |
| LMA (Linear MHA) | Yes | Yes | No |
| Trifast | No | No | Yes |

---

## 8. Reference

### Basic Triangle Attention Layer

```julia
"""
    TriangleAttention(c_in, c_hidden, no_heads; starting=true, inf=1e9f0)

Triangle Attention layer following Algorithm 12 from AlphaFold2.
"""
struct TriangleAttention{P, K, V, O, LN, L, S}
    query_proj::P      # Dense(c_in => c_in) for Q
    key_proj::K       # Dense(c_in => c_in) for K
    value_proj::V      # Dense(c_in => c_in) for V
    output_proj::O     # Dense(c_in => c_in) for output
    bias_linear::L    # Dense(c_in => no_heads) for triangle bias
    layer_norm::LN    # LayerNorm
    starting::S        # Bool: starting or ending node
    inf::Float32
end

function TriangleAttention(
    c_in::Int, c_hidden::Int, no_heads::Int;
    starting::Bool=true, inf::Float32=1e9f0
)
    query_proj = Dense(c_in => c_in)
    key_proj = Dense(c_in => c_in)
    value_proj = Dense(c_in => c_in)
    output_proj = Dense(c_in => c_in)
    bias_linear = Dense(c_in => no_heads; use_bias=false)
    layer_norm = LayerNorm(c_in)
    
    return TriangleAttention(
        query_proj, key_proj, value_proj, output_proj,
        bias_linear, layer_norm, starting, inf
    )
end
```

### Forward Pass

```julia
function (layer::TriangleAttention)(x, mask=nothing)
    # x: [N, N, c_in, batch] in Julia (or [*, N, N, c_in] with *)
    sz = size(x)
    N, N2, c_in, batch = sz[1], sz[2], sz[3], sz[4]
    
    # Transpose for ending node
    if !layer.starting
        x = permutedims(x, (2, 1, 3, 4))  # swap i and j
        mask = permutedims(mask, (2, 1, 3))
    end
    
    # Apply layer norm
    x_norm = layer.layer_norm(x)
    
    # Compute triangle bias
    bias = layer.bias_linear(x_norm)  # [N, N, no_heads, batch]
    bias = permutedims(bias, (3, 1, 2, 4))  # [no_heads, N, N, batch]
    
    # Compute mask bias (OpenFold style)
    if mask !== nothing
        # mask_bias = inf * (mask - 1)
        mask_bias = layer.inf .* (mask .- 1)
        # Reshape for broadcasting: [N, N, batch] → [1, N, 1, 1, N, batch]
        mask_bias = reshape(mask_bias, (1, N, 1, 1, N, batch))
    else
        mask_bias = nothing
    end
    
    # Q, K, V projections
    q = layer.query_proj(x_norm)
    k = layer.key_proj(x_norm)
    v = layer.value_proj(x_norm)
    
    # Apply attention
    out = trifast_attention(q, k, v, bias, mask_bias)
    
    # Output projection
    out = layer.output_proj(out)
    
    # Transpose back for ending node
    if !layer.starting
        out = permutedims(out, (2, 1, 3, 4))
    end
    
    return out
end
```

### Trifast Kernel

```julia
"""
    trifast_kernel!(out, q, k, v, bias, mask)

Core trifast kernel. See src/models/common/attention/trifast.jl for full implementation.
"""
function trifast_kernel!(out, q, k, v, bias, mask=nothing)
    D, H, N, _, Batch = size(q)
    scale = 1.0f0 / sqrt(D)
    inf_val = min(floatmax(Float32) / 2, 1e9f0)
    
    @inbounds for b in 1:Batch, h in 1:H, i in 1:N, j in 1:N
        m_curr = typemin(Float32)
        
        # Pass 1: Find row maximum
        for k_idx in 1:N
            mask_val = (mask === nothing) ? Float32(1) : Float32(mask[i, k_idx, b])
            if iszero(mask_val)
                score = -inf_val
            else
                score = zero(Float32)
                for d in 1:D
                    score += q[d, h, i, j, b] * k[d, h, i, k_idx, b]
                end
                score = score * scale + bias[h, j, k_idx, b]
            end
            m_curr = max(m_curr, score)
        end
        
        # Pass 2: Exponentiate and accumulate
        l_curr = zero(Float32)
        for k_idx in 1:N
            mask_val = (mask === nothing) ? Float32(1) : Float32(mask[i, k_idx, b])
            if !iszero(mask_val)
                score = zero(Float32)
                for d in 1:D
                    score += q[d, h, i, j, b] * k[d, h, i, k_idx, b]
                end
                score = score * scale + bias[h, j, k_idx, b]
                p = exp(score - m_curr)
                l_curr += p
                for d in 1:D
                    out[d, h, i, j, b] += p * v[d, h, i, k_idx, b]
                end
            end
        end
        
        # Pass 3: Normalize
        if l_curr > zero(Float32)
            inv_l = one(Float32) / l_curr
            for d in 1:D
                out[d, h, i, j, b] *= inv_l
            end
        end
    end
    
    return out
end
```

### Type Aliases

```julia
const TriangleAttentionStartingNode = TriangleAttention
TriangleAttentionEndingNode(args...; kwargs...) = 
    TriangleAttention(args...; starting=false, kwargs...)
```

---

## 8. Reference

### Files

| Implementation | File |
|---------------|------|
| OpenFold1 | `python/openfold/openfold/model/triangular_attention.py` |
| OpenFold3 | `python/openfold-3/openfold3/core/model/layers/triangular_attention.py` |
| Boltz2 | `python/boltz/src/boltz/model/layers/triangular_attention/attention.py` |
| Boltz2 Primitives | `python/boltz/src/boltz/model/layers/triangular_attention/primitives.py` |

### Algorithms

| Algorithm | Source | Description |
|-----------|--------|-------------|
| Algorithm 12 | AlphaFold2 | Generic triangle attention layer |
| Algorithm 13 | AlphaFold2 | Triangle attention starting node |
| Algorithm 14 | AlphaFold2 | Triangle attention ending node |
| Algorithm 15 | OpenFold3 | OpenFold3 variant |

### Papers

- [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2) — Algorithms 12-14
- [OpenFold](https://www.biorxiv.org/content/10.1101/2022.11.21.517277v2)
- [Boltz-1](https://www.biorxiv.org/content/10.1101/2024.11.04.621668v1)
- [OpenFold3](https://arxiv.org/abs/2503.19915)

### Julia Implementation

| File | Description |
|------|-------------|
| `src/models/common/attention/trifast.jl` | Reference trifast implementation |
| `TriangleAttention/trifast.md` | Trifast optimization guide |
| `TriangleAttention/flash_attention.md` | FlashAttention-4 innovations |
