# Triangle Multiplication: A Comparative Analysis

> Detailed comparison of Triangle Multiplication implementations in AlphaFold2, Boltz, and OpenFold. Clarifies what differs, what stays the same, and whether efficient kernels are needed.

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Projection Patterns](#3-projection-patterns)
4. [Software Architecture: Static Dispatch](#4-software-architecture-static-dispatch)
5. [Algorithm Anatomy](#5-algorithm-anatomy)
6. [Comparative Analysis](#6-comparative-analysis)
7. [Julia Implementation Strategy](#7-julia-implementation-strategy)
8. [Reference](#8-reference)

---

## 1. TL;DR: What Differs?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXACT DIFFERENCES SUMMARY                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ✅ SAME ACROSS ALL IMPLEMENTATIONS:                                    │
│     - Mathematical operation (triangular einsum)                         │
│     - Outgoing: Σ_k A[i,k] · B[k,j]                                    │
│     - Incoming: Σ_k A[k,i] · B[k,j]                                     │
│     - LayerNorm + gating structure                                       │
│                                                                          │
│  ❌ DIFFERENT ACROSS IMPLEMENTATIONS:                                    │
│     - Projection structure (separate vs fused linear layers)             │
│     - Weight initialization                                             │
│     - Kernel support (CueEquiv, chunked inference)                       │
│                                                                          │
│  CONSEQUENCE:                                                           │
│     - Outputs differ due to different weights/init                        │
│     - But operation semantics are identical                              │
│     - Any correct implementation produces valid model                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

| **Einsum operation** | `bikd,bjkd→bijd` | `bikd,bjkd→bijd` | `bikd,bjkd→bijd` |
| **Projection** | Separate A/B | Separate A/B | **Fused** |
| **Output gating** | Yes | Yes | Yes |
| **LayerNorm** | Yes | Yes | Yes |
| **Chunking support** | Yes | Yes | No |
| **CueEquiv kernels** | Yes | Yes | Yes |

### Implementation Parity Audit

| Component | OpenFold1 | OpenFold3 | Boltz2 |
| :--- | :--- | :--- | :--- |
| **Mult. Projections** | Separate (a/b) | Separate (a/b) | **Fused (ab_p, ab_g)** |
| **Weight Splitting** | N/A | N/A | `torch.chunk` |
| **Output Gating** | Sigmoid | Sigmoid | Sigmoid |
| **Normalization** | LayerNorm (AF2) | LayerNorm (AF3) | LayerNorm (AF3 Style) |

> [!TIP]
> **Boltz2 Optimization**: By fusing the queries for $A$ and $B$, Boltz2 reduces the number of matrix multiplications, which is beneficial for memory bandwidth during the projection phase.

---

## 2. Mathematical Foundations

### The Core Operation

Triangle Multiplication (also known as Triangular Multiplicative Update) updates the relationship between two residues $(i, j)$ by aggregating information from a third residue $k$. This effectively mimics a specialized convolution over the triangle formed by $(i, j, k)$.

#### Outgoing Variant (Algorithm 11)

In the outgoing variant, the update for pair $(i, j)$ combines information from row $i$ and column $j$ that "starts" at residue $i$ and $j$ respectively, meeting at residue $k$.

$$
\begin{aligned}
\mathbf{a}_{ik} &= \sigma(W_{g,a} \mathbf{z}_{ik}) \odot (W_{p,a} \mathbf{z}_{ik}) \\
\mathbf{b}_{kj} &= \sigma(W_{g,b} \mathbf{z}_{kj}) \odot (W_{p,b} \mathbf{z}_{kj}) \\
\mathbf{z}'_{ij} &= \text{LayerNorm}\left(\sum_k \mathbf{a}_{ik} \odot \mathbf{b}_{kj}\right)
\end{aligned}
$$

**Visualizing Outgoing Flow:**
```text
          residue j
             ▼
    ┌───┬───┬───┬───┐
    │   │   │ b │   │  Input: b[k,j]
    ├───┼───┼───┼───┤
 i ─┤ a │   │ z'│   │  Output: z'[i,j] = Σ_k a[i,k] * b[k,j]
    └───┴───┴───┴───┘
          ▲
      Input: a[i,k]
```

#### Incoming Variant (Algorithm 12)

In the incoming variant, the update for pair $(i, j)$ combines information from column $i$ and column $j$ that "ends" at residue $i$ and $j$.

$$
\begin{aligned}
\mathbf{a}_{ki} &= \sigma(W_{g,a} \mathbf{z}_{ki}) \odot (W_{p,a} \mathbf{z}_{ki}) \\
\mathbf{b}_{kj} &= \sigma(W_{g,b} \mathbf{z}_{kj}) \odot (W_{p,b} \mathbf{z}_{kj}) \\
\mathbf{z}'_{ij} &= \text{LayerNorm}\left(\sum_k \mathbf{a}_{ki} \odot \mathbf{b}_{kj}\right)
\end{aligned}
$$

**Visualizing Incoming Flow:**
```text
      residue i residue j
          ▼         ▼
    ┌───┬───┬───┬───┐
    │   │ a │   │ b │  Inputs: a[k,i], b[k,j]
    ├───┼───┼───┼───┤
    │   │   │   │   │
    ├───┼───┼───┼───┤
    │   │ z'│   │   │  Output: z'[i,j] = Σ_k a[k,i] * b[k,j]
    └───┴───┴───┴───┘
```

### Why Triangle Multiplication?

Unlike standard 1D context (sequences), 2D context (pair matrices) requires updates that respect the underlying graph structure of the protein.

- **Outgoing:** Residue $i$ and $j$ interact through an intermediary $k$.
- **Incoming:** Residue $i$ and $j$ are both being interacted with by residue $k$.
- **Complexity:** Both variants are $O(N^3)$, making them the primary computational bottleneck for long sequences.

## 3. Projection Patterns

Triangle Multiplication implementations differ primarily in how they project the pair representation into the input spaces for the multiplicative operation.

### Outgoing Node Projections
...

### OpenFold1/OpenFold3: Separate Projections

```
┌─────────────────────────────────────────────────────────────────────────┐
│              OPENFOLD SEPARATE PROJECTIONS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  4 independent linear layers:                                           │
│                                                                          │
│  z ──► [linear_a_p] ──► A_p                                            │
│    │                                                                     │
│    └──► [linear_a_g] ──► A_g ──► sigmoid(A_g) ◯ A_p ──► A              │
│                                                                          │
│  z ──► [linear_b_p] ──► B_p                                            │
│    │                                                                     │
│    └──► [linear_b_g] ──► B_g ──► sigmoid(B_g) ◯ B_p ──► B              │
│                                                                          │
│  Each layer has independent weights:                                     │
│  - linear_a_p: [C_z, C_hidden]                                           │
│  - linear_a_g: [C_z, C_hidden]                                           │
│  - linear_b_p: [C_z, C_hidden]                                           │
│  - linear_b_g: [C_z, C_hidden]                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Boltz2: Fused Projections

```
┌─────────────────────────────────────────────────────────────────────────┐
│              BOLTZ2 FUSED PROJECTIONS                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  2 fused linear layers (produce 2x channels):                           │
│                                                                          │
│  z ──► [p_in: C_z → 2*C_hidden] ──► [A_p │ B_p] ──► split ──► A, B   │
│    │                                                                     │
│    └──► [g_in: C_z → 2*C_hidden] ──► [A_g │ B_g] ──► sigmoid ◯ ──►   │
│                                                                          │
│  Same parameters, single forward pass, then split:                        │
│  - p_in: [C_z, 2*C_hidden] (weights fused)                             │
│  - g_in: [C_z, 2*C_hidden] (weights fused)                             │
│                                                                          │
│  Memory/compute efficient for inference                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Are They Mathematically Equivalent?

**YES** - Both compute:
$$A = \sigma(W_g^A Z) \cdot W_p^A Z$$
$$B = \sigma(W_g^B Z) \cdot W_p^B Z$$

The fused version just concatenates $W_p^A$ and $W_p^B$ into a single weight matrix, then splits the output.

## 4. Software Architecture: Static Dispatch

To manage the different variants of Triangle Multiplication (Algorithms 11 and 12) without runtime branching, we utilize `Static.jl` to lift configurations to the type level.

### Displacement via Multiple Dispatch

Using `StaticBool`, we can generate specialized methods for the Outgoing and Incoming node variants.

```julia
using Static: static, True, False

function forward(obj::TriangleMultiplication, z)
    # Lift the 'outgoing' field to the type level
    is_outgoing = static(obj.outgoing)
    
    # Selection via dispatch
    return _forward_logic(is_outgoing, obj, z)
end

# Optimization for Algorithm 11 (Outgoing)
function _forward_logic(::True, obj, z)
    # 1. LayerNorm
    # 2. Fused/Separate Projections
    # 3. _triangle_multiplication_kernel(..., ::Outgoing)
end

# Optimization for Algorithm 12 (Incoming)
function _forward_logic(::False, obj, z)
    # 1. LayerNorm
    # 2. Fused/Separate Projections
    # 3. _triangle_multiplication_kernel(..., ::Incoming)
end
```

### Backend-Aware Master Kernels

The master kernel dispatches to the most efficient implementation based on the array type, allowing seamless execution on both CPU and GPU.

```julia
# MASTER DISPATCH
_triangle_multiplication_kernel(a, b, ::AbstractArray) = 
    _triangle_multiplication_kernel_cpu(a, b)

_triangle_multiplication_kernel(a, b, ::CuArray) = 
    _triangle_multiplication_kernel_cuda(a, b)
```

---

## 5. Comparative Code Analysis

### OpenFold1 (Separate)

```python
class TriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    def __init__(self, c_z, c_hidden, _outgoing=True):
        # Separate projection layers
        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
    
    def forward(self, z, mask=None, ...):
        z = self.layer_norm_in(z)
        
        # Separate gated projections
        a = self.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = self.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)
        
        # Apply mask
        a = a * mask.unsqueeze(-1)
        b = b * mask.unsqueeze(-1)
        
        # Triangular multiply (THIS IS IDENTICAL)
        x = self._combine_projections(a, b)  # einsum("bikd,bjkd->bijd", a, b)
        
        # Output
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        x = x * self.sigmoid(self.linear_g(z))
        
        return x
```

### Boltz2 (Fused)

```python
class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, dim: int = 128):
        # Fused projection layers (output 2x channels)
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)   # Fused A+B
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)   # Fused A+B
        
        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x, mask, use_kernels=False):
        if use_kernels:
            return kernel_triangular_mult(...)  # CueEquiv kernel
        
        # Input gating (fused)
        x = self.norm_in(x)
        x = self.p_in(x) * self.g_in(x).sigmoid()
        
        # Apply mask
        x = x * mask.unsqueeze(-1)
        
        # Split and triangular multiply
        a, b = torch.chunk(x.float(), 2, dim=-1)
        
        # Triangular multiply (THIS IS IDENTICAL)
        x = torch.einsum("bikd,bjkd->bijd", a, b)
        
        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x).sigmoid()
        
        return x
```

### Key Code Differences

| Line | OpenFold1 | Boltz2 |
|------|-----------|--------|
| Projection | `linear_a_p(z)`, `linear_b_p(z)` | `p_in(z)` → split |
| Gating | `sigmoid(linear_a_g(z))` | `g_in(x).sigmoid()` |
| Mask | Applied to A and B separately | Applied before split |
| Einsum | `self._combine_projections(a, b)` | `torch.einsum("bikd,bjkd->bijd", a, b)` |

---

## 5. Are Outputs Identical?

### Short Answer: YES, with proper weight mapping

```
┌─────────────────────────────────────────────────────────────────────────┐
│              OUTPUT IDENTICAL GIVEN SAME PARAMETERS                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  All implementations compute the SAME operation:                         │
│     Z_new[i,j] = Σ_k A[i,k] · B[k,j]                                 │
│                                                                          │
│  The einsum is IDENTICAL across all implementations.                     │
│                                                                          │
│  Differences are only in WEIGHT STRUCTURE:                             │
│                                                                          │
│  OpenFold1 ↔ OpenFold3: IDENTICAL weight structure                     │
│     → Outputs identical with same weights                                │
│                                                                          │
│  OpenFold ↔ Boltz2: DIFFERENT weight structure                          │
│     → Outputs identical after weight remapping                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Weight Structure Comparison

| Layer | OpenFold1/OpenFold3 | Boltz2 |
|-------|---------------------|--------|
| `p_in` (input projection) | `linear_a_p` + `linear_b_p` (separate) | `p_in` (fused) |
| `g_in` (input gating) | `linear_a_g` + `linear_b_g` (separate) | `g_in` (fused) |
| `p_out` (output projection) | `linear_z` | `p_out` |
| `g_out` (output gating) | `linear_g` | `g_out` |

### Weight Mapping: OpenFold ↔ Boltz2

**OpenFold → Boltz2 (concatenate):**

```python
# OpenFold separate weights
W_a_p = openfold.linear_a_p.weight   # [C_hidden, C_z]
W_b_p = openfold.linear_b_p.weight   # [C_hidden, C_z]

# Boltz fused weights
boltz.p_in.weight = torch.cat([W_a_p, W_b_p], dim=0)  # [2*C_hidden, C_z]
# Result: [p_in_weight[:C_hidden], p_in_weight[C_hidden:]] = [A_p, B_p]
```

**Boltz2 → OpenFold (split):**

```python
# Boltz fused weights
W_fused = boltz.p_in.weight  # [2*C_hidden, C_z]

# OpenFold separate weights
openfold.linear_a_p.weight = W_fused[:C_hidden]      # First half
openfold.linear_b_p.weight = W_fused[C_hidden:]     # Second half
```

### Same-Parameter Verification

| Comparison | Weights Same? | Structure Same? | Outputs Identical? |
|------------|---------------|----------------|-------------------|
| OpenFold1 ↔ OpenFold3 | ✅ If copied | ✅ Yes | ✅ YES |
| OpenFold ↔ Boltz2 | ✅ If remapped | ❌ No | ✅ YES (after remap) |

### Numerical Precision

Even with correct weight mapping, small differences may arise from:

1. **LayerNorm implementation** - Different epsilon values
2. **Floating point order** - `sigmoid(W*Z)*P` vs `P*sigmoid(W*Z)` (commutative in theory)
3. **Broadcasting** - Mask application order

```python
# These are mathematically equivalent but may differ slightly:
# Option 1: OpenFold style
a = sigmoid(linear_a_g(z)) * linear_a_p(z)
a = a * mask.unsqueeze(-1)

# Option 2: Boltz style  
x = linear_a_g(z) * linear_a_p(z)
x = x * mask  # Broadcasts same way
a = sigmoid(x)
```

Expected numerical difference: < 1e-6 (FP32)
┌─────────────────────────────────────────────────────────────────────────┐
│                    WHEN WOULD OUTPUTS DIFFER?                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. DIFFERENT WEIGHTS                                                   │
│     Each implementation has its own weight initialization:               │
│     - OpenFold1: Standard PyTorch init                                  │
│     - OpenFold3: Custom linear_init_params                              │
│     - Boltz2: Custom init functions (lecun_normal, gating_init, etc.)   │
│                                                                          │
│  2. DIFFERENT PROJECTION STRUCTURE                                       │
│     - Boltz2 uses fused projections (same params for A and B)            │
│     - OpenFold uses separate projections (independent params)              │
│     → Even with same init strategy, weights differ                       │
│                                                                          │
│  3. TRAINING DIFFERENCES                                                 │
│     - Different batch normalization states                                │
│     - Different gradient accumulation patterns                             │
│                                                                          │
│  CONSEQUENCE:                                                            │
│     Models trained with different implementations will have                │
│     DIFFERENT weights and thus DIFFERENT outputs.                         │
│                                                                          │
│  BUT: If you copy weights from one to another, the operation             │
│       is mathematically IDENTICAL.                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### To Verify: Copy Weights

If you copy Boltz weights to OpenFold:

```python
# Copy Boltz weights to OpenFold structure
openfold_model.linear_a_p.weight = boltz.p_in.weight[:dim]
openfold_model.linear_b_p.weight = boltz.p_in.weight[dim:]
# ... etc for all weights

# Now outputs should be IDENTICAL
```

---

## 6. Do We Need Efficient Kernels?

### Short Answer: Probably Not for CPU

The triangular multiply using NNlib's `batched_matmul` is already well-optimized:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              WHY NNlib IS ALREADY OPTIMIZED                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. BLAS-BACKED                                                        │
│     NNlib uses OpenBLAS/MKL/Apple Accelerate                           │
│     → Multi-threaded matrix multiplication                              │
│     → SIMD vectorized                                                  │
│                                                                          │
│  2. MEMORY-BANDWIDTH BOUND                                              │
│     This operation needs to touch: N³ elements                         │
│     → Memory bandwidth is the bottleneck, not compute                   │
│     → Already saturates memory bandwidth with BLAS                      │
│                                                                          │
│  3. O(N³) IS UNAVOIDABLE                                               │
│     The sum over k cannot be avoided                                   │
│     → No algorithmic improvement possible                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Comparison: NNlib vs CueEquiv

| Aspect | NNlib.jl | CueEquiv |
|--------|----------|----------|
| CPU Performance | ✅ Near-optimal | ✅ Near-optimal |
| GPU Performance | Good | ✅ 3-5x faster (fused) |
| Memory | Standard | Fused reduces memory traffic |
| Precision | FP32/FP64 | FP16/BF16 + Tensor Cores |

### When Might Custom Kernels Help?

```
┌─────────────────────────────────────────────────────────────────────────┐
│              WHEN TO WRITE CUSTOM KERNELS                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  GPU: YES                                                               │
│     - Tensor Core utilization (FP16/BF16)                              │
│     - Fused kernel: projections + einsum + gating                        │
│     - Expected speedup: 2-4x                                           │
│                                                                          │
│  CPU: Probably NO                                                       │
│     - Already saturates memory bandwidth                                 │
│     - BLAS is well-optimized                                           │
│     - Custom kernel unlikely to beat MKL/OpenBLAS                       │
│                                                                          │
│  Memory-Constrained: YES (chunking)                                    │
│     - For very long sequences (N > 2000)                                │
│     - Trade compute for memory (like OpenFold's inference mode)          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Summary: Optimization Priority

| Operation | Priority | Notes |
|-----------|----------|-------|
| **Triangle Attention** | HIGH | O(N²) but softmax is compute-bound |
| **Triangle Multiplication** | LOW | Already memory-bandwidth bound, NNlib is fine |

### If You Still Want GPU Kernels

```julia
# GPU optimization would focus on:
# 1. Fusing: LayerNorm → projections → einsum → output
# 2. Tensor Core matmul with FP16
# 3. Reducing memory bandwidth by keeping data in registers

# But expected gain: 2-4x (vs ~100x for Triangle Attention with Trifast)
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPUTATIONAL COMPLEXITY                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Standard matmul:  O(N²)                                                 │
│  Triangle multiply: O(N³)  ← Bottleneck!                               │
│                                                                          │
│  For einsum("bikd,bjkd->bijd", A, B):                                   │
│                                                                          │
│  Dimensions:                                                            │
│    A, B: [B, N, N, D]                                                   │
│                                                                          │
│  Computation:                                                           │
│    For each (b, i, j, d):                                               │
│        sum over k from 1 to N                                           │
│    Total: B × N × N × N × D operations                                  │
│                                                                          │
│  Example: N = 1000, D = 128                                              │
│    FLOPs ≈ 1000 × 1000 × 1000 × 128 = 1.28e11                          │
│    ≈ 128 TFLOPS (per layer!)                                             │
│                                                                          │
│  For comparison:                                                         │
│    AlphaFold2 has ~48 triangular multiply layers                        │
│    Total: ~6,000 TFLOPS per forward pass                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### What Kernels Help With?

| Kernel Type | Speedup | Notes |
|-------------|---------|-------|
| **CueEquiv** | ~3-5x | Fused kernel for A,B projections + einsum |
| **Chunked inference** | 2x memory | Reduces memory, not compute |
| **FP16/BF16** | ~2x | Memory bandwidth bound |

### Do We Need a Trifast-style Kernel?

**For Julia: YES**

Unlike the Python implementations that can use:
- CUDA optimized `torch.einsum`
- CueEquiv fused kernels

Pure Julia with NNlib.jl:
- No dedicated triangular multiply kernel
- Falls back to generic batched operations
- Significant performance gap vs optimized implementations

### Recommended Approach for Julia

```
┌─────────────────────────────────────────────────────────────────────────┐
│              JULIA OPTIMIZATION STRATEGY                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. CPU (Small N < 256):                                               │
│     - Use LoopVectorization.jl for batched matmul                       │
│     - ~2-3x speedup possible                                            │
│                                                                          │
│  2. CPU (Large N > 256):                                               │
│     - Consider chunking similar to OpenFold                             │
│     - Trade compute for memory                                          │
│                                                                          │
│  3. GPU (any N):                                                        │
│     - Write CUDA.jl kernel for triangular multiply                       │
│     - Fuse projection + einsum + gating                                 │
│     - Target: Match CueEquiv performance                                │
│                                                                          │
│  Priority:                                                              │
│     Triangle Attention (trifast) > Triangle Multiplication             │
│     Attention is O(N²) with softmax → more critical                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Julia Implementation Guide

### Core Implementation

```julia
"""
    TriangleMultiplication(dim, hidden_dim; incoming=false)

Triangle Multiplication following Algorithms 7 & 8 from AlphaFold2.
Uses NNlib's batched_matmul which is already optimized.
"""
struct TriangleMultiplication{NI, NO, LAP, LAG, LBP, LBG, LZ, LG, I<:StaticBool}
    dim::Int
    hidden_dim::Int
    incoming::I
    layer_norm_in::NI
    layer_norm_out::NO
    linear_a_p::LAP
    linear_a_g::LAG
    linear_b_p::LBP
    linear_b_g::LBG
    linear_z::LZ
    linear_g::LG
end

function (m::TriangleMultiplication)(msa, mask, ps, st)
    # 1. Input LayerNorm
    x_norm, st_ni = m.layer_norm_in(msa, ps.layer_norm_in, st.layer_norm_in)
    
    # 2. Gated Projections
    a_p = m.linear_a_p(x_norm)
    a_g = sigmoid.(m.linear_a_g(x_norm))
    a = a_p .* a_g
    
    b_p = m.linear_b_p(x_norm)
    b_g = sigmoid.(m.linear_b_g(x_norm))
    b = b_p .* b_g
    
    # 3. Apply Mask
    mask_3d = reshape(mask, 1, size(mask)...)
    a = a .* mask_3d
    b = b .* mask_3d
    
    # 4. Triangular Multiply using NNlib (already optimized)
    z = triangular_multiply(a, b, m.incoming)
    
    # 5. Output
    z = m.layer_norm_out(z)
    z = m.linear_z(z)
    g = sigmoid.(m.linear_g(x_norm))
    out = z .* g
    
    return out, st_new
end
```

### Triangular Multiply (Using NNlib)

```julia
using NNlib

"""
    triangular_multiply(a, b, incoming)

Compute triangular multiply using NNlib.batched_matmul.

Outgoing: einsum("bikd,bjkd->bijd", a, b)  
Incoming: einsum("bkid,bkjd->bijd", a, b)

Julia layout: [D, N, N, B]
Python layout: [B, N, N, D]

NNlib's batched_matmul is already optimized:
- BLAS-backed (OpenBLAS/MKL/Accelerate)
- Multi-threaded
- SIMD vectorized
"""
function triangular_multiply(a::AbstractArray{T,4}, b::AbstractArray{T,4}, 
                            incoming::StaticBool) where T
    # Julia layout: [D, N, N, B]
    # Permute to: [B, N, N, D] for NNlib
    a_b = permutedims(a, (4, 2, 3, 1))  # [B, N, N, D]
    b_b = permutedims(b, (4, 2, 3, 1))  # [B, N, N, D]
    
    if incoming
        # einsum("bkid,bkjd->bijd", a, b)
        # a[b,k,i,d] * b[b,k,j,d] -> z[b,i,j,d]
        z_b = batched_matmul(a_b, permutedims(b_b, (1, 2, 4, 3)))
    else
        # einsum("bikd,bjkd->bijd", a, b)
        # a[b,i,k,d] * b[b,j,k,d] -> z[b,i,j,d]
        z_b = batched_matmul(a_b, permutedims(b_b, (1, 3, 2, 4)))
    end
    
    # Back to Julia layout: [D, N, N, B]
    return permutedims(z_b, (4, 2, 3, 1))
end
```

### Why No Further Optimization Needed?

```
┌─────────────────────────────────────────────────────────────────────────┐
│              PERFORMANCE ANALYSIS                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Operation: einsum("bikd,bjkd->bijd", a, b)                            │
│                                                                          │
│  Memory access: 2 × N³ × D elements (read A, read B)                   │
│  Compute:       2 × N⁴ × D FLOPs (multiply + add)                     │
│                                                                          │
│  Arithmetic Intensity: 2 × N⁴ × D / (2 × N³ × D × 8 bytes)            │
│                       = N / 4 bytes                                    │
│                                                                          │
│  For N=1000: AI ≈ 250 FLOPs/byte → MEMORY BANDWIDTH BOUND              │
│                                                                          │
│  NNlib with BLAS already saturates memory bandwidth.                     │
│  Further optimization unlikely to help.                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Reference

### Files

| Implementation | File |
|---------------|------|
| OpenFold1 | `python/openfold/openfold/model/triangular_multiplicative_update.py` |
| OpenFold3 | `python/openfold-3/openfold3/core/model/layers/triangular_multiplicative_update.py` |
| Boltz2 | `python/boltz/src/boltz/model/layers/triangular_mult.py` |

### Julia Implementation

| File | Description |
|------|-------------|
| `src/models/common/attention/triangle_multiplication.jl` | Reference Julia implementation |
| `TriangleAttention/triangle_attention.md` | Triangle Attention documentation |
| `TriangleAttention/trifast.md` | Triangle Attention optimization guide |

### Papers

- [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2) — Algorithms 7-12
- [Boltz-1](https://www.biorxiv.org/content/10.1101/2024.11.04.621668v1)
- [OpenFold3](https://arxiv.org/abs/2503.19915)
