# 1. Layers, masks and bias

```@meta
CurrentModule = LuxTriangleAttention
```

The package has three layers you use directly, and two small ones they are built from:

| layer | what | input |
|:--|:--|:--|
| [`TriangleAttention`](@ref) | starting / ending node attention with a triangle bias (AlphaFold2 Algorithms 13, 14) | `[C, Ni, Nj, B]` |
| [`TriangleMultiplication`](@ref) | outgoing / incoming triangle multiplication (AlphaFold2 Algorithms 11, 12) | `[C, N, N, B]` |
| [`Attention`](@ref) | multi-head attention with optional gate, pair bias and mask | `[C, N, B]`, `[C, N, S, B]` or a tuple |
| [`TriMulCore`](@ref) | the projections and contraction inside `TriangleMultiplication` | `[C, N, N, B]` |
| [`GatedLinearUnit`](@ref), [`SwiGLU`](@ref) | gated projections | `[C, ...]` |

All of them are ordinary Lux layers: `Lux.setup` makes the parameters and state, and a call returns `(y, st)`.

## Triangle attention

```julia
using Lux, LuxTriangleAttention, Random

rng = Random.default_rng()
x = randn(Float32, 64, 32, 32, 1)               # [C, Ni, Nj, B]

attn = TriangleAttention(64, 32, 4; is_starting=true)
ps, st = Lux.setup(rng, attn)
y, st = attn(x, ps, st)                          # [64, 32, 32, 1]
```

The arguments are the input channels, the head dimension and the number of heads. With `is_starting=true` each
`(i, j)` attends over its row (the *starting node*, AlphaFold2 Algorithm 13); with `false` over its column (the
*ending node*, AlphaFold2 Algorithm 14). The ending node transposes `i` and `j` around the same computation.

**AlphaFold2 Algorithm 13**, triangular gated self-attention around the starting node, for each head ``h``:

```math
\begin{aligned}
z_{ij} &\leftarrow \mathrm{LayerNorm}(z_{ij}) \\
q^h_{ij},\, k^h_{ij},\, v^h_{ij} &= \mathrm{LinearNoBias}(z_{ij}) \\
b^h_{ij} &= \mathrm{LinearNoBias}(z_{ij}) \\
g^h_{ij} &= \mathrm{sigmoid}\big(\mathrm{Linear}(z_{ij})\big) \\
a^h_{ijk} &= \mathrm{softmax}_k\Big(\tfrac{1}{\sqrt{c}}\, {q^h_{ij}}^{\top} k^h_{ik} + b^h_{jk}\Big) \\
o^h_{ij} &= g^h_{ij} \odot \sum_k a^h_{ijk}\, v^h_{ik} \\
\tilde{z}_{ij} &= \mathrm{Linear}\big(\mathrm{concat}_h(o^h_{ij})\big)
\end{aligned}
```

**AlphaFold2 Algorithm 14**, around the ending node, differs in lines 5 and 6 only:

```math
\begin{aligned}
a^h_{ijk} &= \mathrm{softmax}_k\Big(\tfrac{1}{\sqrt{c}}\, {q^h_{ij}}^{\top} k^h_{kj} + b^h_{ki}\Big) \\
o^h_{ij} &= g^h_{ij} \odot \sum_k a^h_{ijk}\, v^h_{kj}
\end{aligned}
```

Here ``c`` is the head dimension. The `Linear` layers carry biases unless `use_bias` says otherwise, and the optional
mask (see [Masks](@ref)) is not part of the algorithms: it removes keys from the softmax.

The triangle bias is not an input: the layer projects it from the normalised `x` itself, as the references do, and
hands it to the inner [`Attention`](@ref) with `bias_layout=:qk`. Keyword arguments the layer does not use itself
(`chunk_size`, `use_kernel`, `use_gate`, ...) are forwarded to that [`Attention`](@ref).

## Triangle multiplication

```julia
mul = TriangleMultiplication(64, 128; is_outgoing=true)   # channels, hidden channels
ps, st = Lux.setup(rng, mul)
y, st = mul(x, ps, st)                                    # [64, 32, 32, 1]
```

`is_outgoing=true` is AlphaFold2 Algorithm 11 (triangular multiplicative update using "outgoing" edges) and `false`
is AlphaFold2 Algorithm 12 ("incoming" edges):

```math
\begin{aligned}
z_{ij} &\leftarrow \mathrm{LayerNorm}(z_{ij}) \\
a_{ij},\, b_{ij} &= \mathrm{sigmoid}\big(\mathrm{Linear}(z_{ij})\big) \odot \mathrm{Linear}(z_{ij}) \\
g_{ij} &= \mathrm{sigmoid}\big(\mathrm{Linear}(z_{ij})\big) \\
\tilde{z}_{ij} &= g_{ij} \odot \mathrm{Linear}\Big(\mathrm{LayerNorm}\big(\textstyle\sum_k a_{ik} \odot b_{jk}\big)\Big)
\qquad \text{(AlphaFold2 Algorithm 11, outgoing)}
\end{aligned}
```

AlphaFold2 Algorithm 12 (incoming) sums over the first index instead, ``\sum_k a_{ki} \odot b_{kj}``; everything else is the same.
The projections run token-major (``xᵀ Wᵀ``) so that the
contraction is a batched matrix product on contiguous `[N, N]` planes without a `permutedims`; the parameters are
those of the plain `Dense` layers (see [`ChannelsLast`](@ref)), so weights from a PyTorch reference load the usual way.

## Masks

A mask is `true` where the position is **kept**. Pass inputs as a `NamedTuple` with `x` and `mask`:

```julia
mask = rand(Bool, 32, 32, 1)                     # [Ni, Nj, B]
y, st = attn((; x, mask), ps, st)
y, st = mul((; x, mask), ps, st)
```

For [`TriangleAttention`](@ref) and [`TriangleMultiplication`](@ref) the mask is the pair mask `[Ni, Nj, B]`. A bare
[`Attention`](@ref) also takes a sequence mask, and [`prep_mask`](@ref) reshapes either to the internal format:

| given | becomes |
|:--|:--|
| `[N, B]` | `[N, 1, 1, B]` |
| `[Ni, Nj, B]` (or `[N, S, B]`) | `[Ni, 1, 1, Nj, B]` |

```julia
model = Attention(64, 32, 4)                     # channels, head dimension, heads
ps, st = Lux.setup(rng, model)
x = randn(Float32, 64, 32, 1)                    # [C, N, B]
mask = rand(Bool, 32, 1)                         # [N, B]
y, st = model((; x, mask), ps, st)
```

!!! note "A query with no live key"
    If every key of a row is masked, the query attends **uniformly** and returns the mean of `v`, as in OpenFold and
    Boltz2 (whose finite `1e9·(mask - 1)` bias makes all logits equal). It is never `NaN`, in any precision, on any
    path including the kernels. See [`SDPA`](@ref).

## Bias layouts

[`Attention`](@ref) takes an optional bias of shape `[H, Nq, Nk, B]` (the PyTorch pair-bias layout) in the
`bias` field. [`prep_bias`](@ref) permutes it into the internal `[Nk, Nq, H, 1, B]` format, according to the
layer's `bias_layout`:

- **`:qk`**, used by [`TriangleAttention`](@ref). For an input `[C, Ni, Nj, B]` and a bias `[H, Ni, Nj, B]` the bias
  becomes `[Ni, Nj, H, 1, B]`. The input is permuted too (for the ending node), so the bias needs no transpose.
- **`:kq`** (the default), matching `AttentionPairBias` of the AlphaFold3 and Boltz references. For `[C, N, S, B]`
  and a bias `[H, Nq, Nk, B]` it becomes `[Nk, Nq, H, 1, B]`, the transpose that accounts for Julia attending over
  dimension 3 where Python attends over dimension 4.

```julia
using Static: static

x = randn(Float32, 16, 32, 32, 1)
bias = randn(Float32, 4, 32, 32, 1)

prep_bias(bias, x, static(:qk))                  # [Ni, Nj, H, 1, B]
prep_bias(bias, x, static(:kq))                  # [Nj, Ni, H, 1, B]

model = Attention(16, 8, 4; bias_layout=:kq)
ps, st = Lux.setup(rng, model)
y, st = model((; x, bias), ps, st)               # prep_bias is applied for you
```

(`prep_bias` takes the layout as a static symbol, which is how the layer stores it.)

## Cross-attention and the fused projection

Pass a tuple `(q_input, kv_input)` or `(q, k, v)` for cross-attention:

```julia
cross = Attention(64, 48, 48, 32, 4)             # chn_q, chn_k, chn_v, head_dim, heads
ps, st = Lux.setup(rng, cross)
xq, xkv = randn(Float32, 64, 16, 1), randn(Float32, 48, 24, 1)
y, st = cross((xq, xkv), ps, st)                 # [64, 16, 1]
```

`fuse_qkv=true` (the default) uses one `Dense` for `q`, `k` and `v` when their channels match, and one for `k` and
`v` when only those match. `use_gate=false` removes the sigmoid output gate, and `use_bias` takes a `Bool` or a
`NamedTuple` over `(:qkv, :q, :kv, :k, :v, :gate, :out)` (see [`resolve_defaults`](@ref)).

## Reducing the score tensor

The plain attention path materialises the `[Nq, Nk, H, Nj, B]` scores, which is O(N³·H) for a pair input
(about 1 GiB per op at N = 400). Two options keep that bounded, and they are alternatives (combining them throws):

- `use_kernel=true`: a fused kernel that never materialises the scores. See
  [Fused attention kernels](2-fused-kernels.md).
- `chunk_size=n`: attend over slices of `n` along dimension 4. See [Compiling with Reactant](3-reactant.md).

## Verifying against the references

The test suite compares every layer to the PyTorch implementations of AlphaFold2, AlphaFold3 and Boltz2, across
precisions, masks, bias layouts and chunk sizes. `test/python/` sets up the reference environment.
