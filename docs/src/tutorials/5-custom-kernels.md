# 5. Your own attention kernel

```@meta
CurrentModule = LuxTriangleAttention
```

You have compiled triangle-attention kernels, a `.metallib` or a `.cubin`/`.ptx`, with a forward and a backward,
possibly spread over several files. This is how to make `TriangleAttention(...; use_kernel=true)` run yours instead of
the built-in ones.

The machinery is [KernelDispatch.jl](https://github.com/Janssena/KernelDispatch.jl), whose documentation explains the
concepts used here (ops, kernels, variants, [`call_binary`](https://janssena.github.io/KernelDispatch.jl/dev/api/)).
In short: attention is the op [`SDPA`](@ref), a kernel is a type plus a few methods, and KernelDispatch passes the
arguments **you** give it straight to your binary, in the order you choose, in plain Julia.

```julia
using LuxTriangleAttention
use_kernel!(:trifast)                                          # the next top-level statement on
KernelDispatch.forward(SDPA(), Trifast(), q, k, v, mask, bias) # or one call with it, no global switch
```

## 1. A worked example: trifast

[`test/support/trifast_kernel.jl`](https://github.com/Janssena/LuxTriangleAttention.jl/blob/main/test/support/trifast_kernel.jl)
defines [trifast](https://github.com/latkins/trifast)'s kernel and
[`test/bench/kernels/registry_setup.jl`](https://github.com/Janssena/LuxTriangleAttention.jl/blob/main/test/bench/kernels/registry_setup.jl)
registers its four Triton metallibs. It has a forward, then a backward split three ways: `dq` (which also writes
`delta`), then `dkv`, then a dedicated `dbias`. Abridged:

```julia
import KernelDispatch
using KernelDispatch: KernelBinary, OutArray, call_binary

struct Trifast <: KernelDispatch.AbstractKernel end
KernelDispatch.kernelname(::Trifast) = :trifast

function KernelDispatch.forward(op::SDPA, k::Trifast, q0, k0, v0, mask, bias)
    q, kk, v, m, b, d = LuxTriangleAttention._sdpa_prep(q0, k0, v0, mask, bias, false)
    o, lse = call_binary(op, k, :fwd,
        OutArray(d.T, size(q)), OutArray(Float32, 1, d.Nq, d.H, d.Nj, d.B),    # outputs
        q, kk, v, b, m, scale, neg_inf, Int32(d.Nq), Int32(d.H);               # inputs, then scalars
        key=(nameof(d.T), d.hd), extent=(d.Nq, d.Nq, d.H * d.B))               # which variant; the problem
    return LuxTriangleAttention._finish(o, lse, q0)
end

KernelDispatch.backward(op::SDPA, k::Trifast, (q0, k0, v0, mask, bias), (o, lse), (dO, _)) = ...
#   three `call_binary`s: (:dq → delta, dq), (:dkv → dk, dv), (:dbias → dbias, zero-filled);
#   returns (dq, dk, dv, nothing, dbias) in the arguments' shapes and element types

KernelDispatch.register_kernel!(SDPA(), Trifast())
KernelDispatch.add_variant!(SDPA(), Trifast(), (:Float32, 32);                 # variant key: (dtype, dim)
    fwd=KernelBinary("_fwd.metallib"; threadgroup=1024, tile=(32, 1, 1)),      # 32 query rows per group
    dq=KernelBinary("_bwd_q.metallib"; threadgroup=1024, tile=(32, 1, 1)),
    dkv=KernelBinary("_bwd_kv.metallib"; threadgroup=1024, tile=(32, 1, 1)),
    dbias=KernelBinary("_bwd_b.metallib"; threadgroup=1024, tile=(32, 32, 1))) # 32×32 pair blocks
```

- `KernelBinary`s are copied into the kernel cache and recorded in a manifest, so registering is a one-time step and
  survives a restart. The cache is KernelDispatch': `~/.cache/KernelDispatch/ops/LuxTriangleAttention.sdpa/<kernel>/`
  (`KERNELDISPATCH_CACHE` moves the root).
- The entry point looked up in each binary is the file's basename, unless you pass `entry=`.
- `call_binary` binds its arguments **in the order given**: each `OutArray(T, dims...)` is a fresh output
  (`zero=true` zero-fills it, for a kernel that accumulates) and is returned, in order; each array is bound in place;
  each number is bound by value.
- **Launch size.** `call_binary` launches `cld.(extent, tile)` threadgroups: the call's `extent` (the problem size, here
  `(Nq, Nq, H·B)`) over each binary's `tile` (how much one threadgroup covers, a property of the build). Trifast thus
  gets `(n/32, n, H·B)` and, for `:dbias`, `(n/32, n/32, H·B)`. There is no `grid` method to write.
- Under Reactant with a Metal client the same call becomes a custom call inside the compiled program.

`LuxTriangleAttention._sdpa_prep` and `_finish` are internal helpers that turn the layer's operands into the form
every built-in kernel reads, and back. They are the easiest way to get the operands right, but they are not API:
check them against your version. What they produce is on the [Kernel operands](../kernel-operands.md) page.

## 2. Declare the types

A binary takes raw pointers, so a `Float16` array handed to a kernel compiled for `float*` would run and compute
something else. KernelDispatch therefore checks the types of a launch against the `signature` the binary was built
for and refuses a mismatch, naming the argument. A prebuilt binary declares it:

```julia
F, M = Vector{Float32}, Vector{Bool}
KernelBinary("_fwd.metallib"; threadgroup=1024, tile=(32, 1, 1),
    signature=(F, F, F, F, F, F, M, Float32, Float32, Int32, Int32))   # o lse q k v bias mask | scale neg_inf n h
```

One entry per bound argument, outputs included, in order: `Vector{T}` for any array, the exact type for a scalar
(`Int32(n)`, not `n`). Without one the first launch warns once and prints the `signature=` that fits. The *order* of
two arguments of the same type is **not** checked, which is what the next section is for.

Float32, Float16 and BFloat16 buffers are supported, and one launch may mix them (a `Float16` `o` beside a `Float32`
`lse`). The built-in kernels take `q`, `k`, `v`, `dO` and the bias in the storage type (the op converts a `k`, `v`,
`dO` or bias of another precision to `q`'s) and keep `lse`, `delta` and `dbias` in Float32.

## 3. Pin your argument lists

Nothing checks that the argument order matches your binary: a mis-ordered list binds without error and computes
garbage. Record what your `forward` and `backward` bind with `KernelDispatch.record_bindings` (each record has a
`label`, the `slots` and the threadgroups `groups`) and compare it in a test, slots and grids; a mis-ordered list also shows up as results that differ from a
reference implementation.

## 4. Variants

`variant_key` (or `call_binary(...; key=...)`) says which build serves a call. A third-party kernel compiled once for
every length keys on `(dtype, dim)`, like trifast. The built-in kernels key on `(dtype, dim, N, hasbias, hasmask)`: a
call by its own length `N`, a variant by the length it was tuned for, with
[`nearest`](https://janssena.github.io/KernelDispatch.jl/dev/api/) picking the closest tuned length (absolute
distance, ties to the larger) but never a different dtype, dimension or presence. Their `build!` tunes the default
lengths lazily when there is no variant at all. See [Fused attention kernels](2-fused-kernels.md#Tuning).

## 5. Differentiation

`backward(op, kernel, args, outs, cotangents)` is what makes a kernel differentiable. Two things call it:

- **Enzyme**, eagerly on device arrays, through KernelDispatch' one rule on `forward`. Enzyme + Metal is not a
  training stack: the rule's gradients are exact, but Enzyme cannot differentiate the ordinary Metal.jl code
  around it. Its payoff is CUDA.
- the layer-level reverse helper, eager or traced: on a trace it emits the backward custom calls.

A gradient traced *through* a kernel under Reactant throws until Enzyme-JAX can differentiate a custom call
([EnzymeAD/Enzyme#2516](https://github.com/EnzymeAD/Enzyme/issues/2516)). Use `use_kernel=false` for differentiated
traced programs.

## 6. Other backends and operations

A prebuilt binary belongs to one backend, so a kernel that must run on both Metal and CUDA ships a binary for each, or
is written in KernelAbstractions and compiled with `KernelDispatch.ka_compile` for whichever backend is loaded.
On CUDA, a `.cubin` (built for one architecture, which `target` records) or a portable `.ptx` takes the place of the
`.metallib`:

```julia
KernelBinary("fwd.cubin"; entry="attn_fwd", threadgroup=128, tile=(128, 1, 1), target="sm_86")
```

Anything beyond attention gets its own KernelDispatch op the same way: an `AbstractKernelOp`, its kernels, and their
`forward` and `backward`. See the [KernelDispatch tutorials](https://janssena.github.io/KernelDispatch.jl/dev/).

Not supported: `.so` / torch-extension kernels (DeepSpeed4Science, cuEquivariance) ship no raw binary with a
documented argument list, so they need a C-call mechanism that does not exist here.
