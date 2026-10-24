# Kernel operands

```@meta
CurrentModule = LuxTriangleAttention
```

What an attention kernel reads, for anyone [writing one](tutorials/5-custom-kernels.md) or reading the built-in ones.
The operands are handed over **in place**: nothing is permuted, padded or copied beyond what this page says.

Notation: `hd` is the head dimension, `H` the heads, `Nq` the attention length (queries = keys), `Nj` the outer
extent (`== Nq` for triangle attention, `1` for a plain `[hd, H, N, B]` sequence), `B` the batch and `T` the storage
type (`Float32`, `Float16` or `BFloat16`; BFloat16s.jl's, the type Metal.jl and Reactant use). Every array is
**column-major, first axis fastest**.

| operand | shape | eltype | notes |
|:--|:--|:--|:--|
| `q`, `k`, `v`, `dO`, `o` | `[hd, H, Nq, Nj, B]` | `T` | offset `d + hd·(h + H·(j + Nq·(i + Nj·b)))` |
| `lse`, `delta` | `[1, Nq, H, Nj, B]` | `Float32` | offset `j + Nq·(h + H·(i + Nj·b))`; natural-log `logsumexp`, and `Σ_d o·dO` |
| `bias` | `[Nk, Nq, H, B]` | `T` | keys fastest; shared across `Nj` |
| `dbias` | `[Nk, Nq, H, B]` | `Float32` | accumulate into a zeroed `OutArray`; sum over `Nj` |
| `mask` | `[Nk, Nj, B]` | `Bool` (1 byte) | **`true` = DROP the key**: a key mask per outer index. The layer's mask is a KEEP mask; `_sdpa_prep` negates it |

An absent bias or mask becomes a 1-element placeholder when `skip = true`, which is only for a kernel whose binaries
are specialised never to read it (the built-in ones). With `skip = false` it becomes a zero bias and an all-live mask at
full size, which is what trifast takes.

Two things that are silent when wrong:

- **Batch and head are not adjacent** in the `[hd, H, Nq, Nj, B]` arrays or in `lse` and `delta`. A kernel that walks
  a combined `bh = b·H + h` index with one stride is right only for `B = 1`. The bias *is* `(b, h)`-adjacent.
- **Each array and each scalar is its own Metal buffer slot.** A launch binds at most 31 of them, so derive strides in
  the kernel rather than passing one per dimension.

## The built-in argument lists

Written once, in `src/kernels/stage_args.jl` (`stage_args`, used by the op and by the tuner alike), and pinned by
`test/kernels/stage_args.test.jl`. Each launch's threadgroups are
`(cld(Nq, rows), Nj, H·B)` with `rows` the binary's `tile[1]`.

```
:default   fwd  o lse | q k v bias_t mask | scale neg_inf n nouter nh mheads
           dq   dq dbias_t | q k v bias_t mask dO lse delta | scale n nouter nh mheads
           dk   dk | q k v bias mask dO lse delta | scale n nouter nh mheads
           dv   dv | q k bias mask dO lse | scale n nouter nh mheads
:flash_mma fwd  o lse | q k v bias mask | scale n nouter nh mheads
           dkv  dk dv dbias | q k v bias mask dO lse delta | scale n nouter nh mheads
           dq   dq | q k v bias mask dO lse delta | scale n nouter nh mheads
```

Outputs come first, then the inputs, then the scalars: `scale = 1/√hd` (`Float32`), `n = Nq`, `nouter = Nj`, `nh = H`,
`mheads = H` (all `Int32`). `bias_t` and `dbias_t` are the pair bias with its two pair axes swapped (`[Nq, Nk, H·B]`,
queries fastest), which a kernel with one thread per query reads coalesced.

## Variants

The built-in kernels key on `(dtype, dim, N, hasbias, hasmask)`; see
[Fused attention kernels](tutorials/2-fused-kernels.md#Tuning).

!!! note "Internal"
    The helper functions that build these operands (`_sdpa_prep`, `_finish`, `_full`, `_res`, ...) are internal. They
    are documented in their docstrings and are the quickest route to correct operands, but they may change between
    versions.
