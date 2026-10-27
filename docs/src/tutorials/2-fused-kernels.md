# 2. Fused attention kernels

```@meta
CurrentModule = LuxTriangleAttention
```

The ordinary attention path computes the whole score tensor. For a pair representation that is O(N³·H) memory, and
it is the part of a Pairformer block that does not fit at large N. A *fused* kernel tiles the keys inside the GPU
kernel (the FlashAttention scheme) and never materialises the scores. This package ships two of them, for Apple GPUs (Metal)
and NVIDIA GPUs (CUDA), and the means to tune and replace them.

## Turning it on

```julia
using Lux, LuxTriangleAttention, Metal, Random

attn = TriangleAttention(128, 32, 4; use_kernel=true)
ps, st = Lux.setup(Random.default_rng(), attn)

dev = Lux.gpu_device()                           # Metal (or CUDA: `using CUDA`), once its package is loaded
ps, st = dev(ps), dev(st)
x = dev(randn(Float32, 128, 256, 256, 1))
y, st = attn(x, ps, st)
```

`use_kernel` is a property of the layer, so it is part of its type and the choice is made at compile time.
It takes effect on **device arrays**. On the host a layer built with `use_kernel=true` runs the ordinary softmax and
gives the same answer, so a model can be built once and run anywhere. It cannot be combined with `chunk_size`, and it
needs self-attention-shaped `q` and `k` (windowed cross-attention with 32 queries against 128 keys is refused with a
message, not mis-addressed).

!!! note "No device kernel, no fallback"
    A device array with no usable kernel is a configuration error and throws. Silently running host attention on a
    GPU would hide a 10–100× slowdown behind a correct answer.

## The two built-in kernels

Attention is one [KernelDispatch](https://github.com/Janssena/KernelDispatch.jl) op, [`SDPA`](@ref), and the
package registers two kernels for it:

| name | what |
|:--|:--|
| `:default` | a KernelAbstractions flash kernel, one thread per query |
| `:flash_mma` | the matrix-unit variant: Apple `simdgroup` matrices, or NVIDIA tensor cores (see [NVIDIA GPUs](#NVIDIA-GPUs)) |

```julia
list_kernels()                 # the kernels, their variant counts and the selected one
current_kernel()               # :default

use_kernel!(:flash_mma)        # this session, from the next top-level statement
current_kernel()               # :flash_mma
```

`use_kernel!` is a *method*, not a table lookup, which keeps the op call type-stable. Two consequences follow from that:

- it takes effect from the **next top-level statement**. An attention call later in the same function or block
  throws instead of silently running the previous kernel;
- to run a single call with a given kernel and no global switch, call `forward` directly:
  `KernelDispatch.forward(SDPA(), LuxTriangleAttention.FlashMMA(), q, k, v, mask, bias)`.

To change what runs by default in a project, persist the choice through KernelDispatch (it is stored as a package
preference and applies from the next session):

```julia
using KernelDispatch
set_default_kernel!(SDPA, :flash_mma)
```

## NVIDIA GPUs

Load CUDA.jl next to this package (`using CUDA`) and both kernels run on the GPU through the same calls as on Metal.
The tensor-core kernel `:flash_mma` uses `mma.sync` (m16n8k16), which exists from Ampere (compute capability 8.0) on; an
older GPU runs `:default`. What differs from Metal:

- **Head dimension.** The tensor cores contract over 16 channels, so `:flash_mma` takes a head dimension that is a
  multiple of 16 (Metal: 8). Other heads, the AlphaFold size 8 among them, run `:default`; measured on an A40, padding 8 to
  16 does not pay in Float16 and Float32 and does in BFloat16.
- **Storage types.** Float16 and BFloat16 run on the tensor cores as they are. Float32 data runs at **Float16**: the
  kernel rounds its inputs to Float16, accumulates in Float32 and returns Float32. The rounding saturates at ±65504 (a
  mask written as a `-1e9` bias stays a mask), and the accuracy is that of Float16 storage, about 1e-3 relative.
  Use `:default` where Float32 must stay exact.
- **Fallback.** A call that selected `:flash_mma` for something it cannot run (a head dimension, a type, a GPU without
  tensor cores) runs `:default` and says so once. `tune_flash_mma!` for such a case does nothing.
- **Shared memory.** KernelAbstractions' local memory is static, which CUDA caps at 48 KiB per block whatever the device
  offers, so the tuner's admissible configurations are bounded by that.
- **Compiling.** A CUDA binary is built without CUDA.jl's runtime state, so that XLA can launch the same file from a
  Reactant trace; a kernel whose code can throw (a bounds check, an integer conversion that can overflow, a division by a
  runtime zero) is refused when it is built. The built-in kernels are written for that: indices are converted with
  truncation, divisions by launch parameters are unchecked, and the group index is decoded by hand.

**Speed.** Measured on an NVIDIA A40 on triangle-attention shapes (4 heads, n = 128–384, head dimensions 16–64), `:flash_mma`
against `:default`, forward and backward:

| storage | forward | backward |
|:--|--:|--:|
| Float16 | 1.5–2.3× | 1.7–2.7× |
| BFloat16 | 2.4–4.4× | 1.9–3.3× |
| Float32 (computed at Float16) | 1.1–1.6× | 1.4–1.6× |

The Float32 rows include rounding the inputs and widening the results in separate passes. These kernels reach roughly 10–17
TFLOP/s, far below the tensor-core peak: they are a first version, written for correctness, and the benchmark that
produced the table is `test/bench/kernels/cuda.bench.jl`.

## Tuning

A kernel's speed depends on the device, the storage type, the head dimension and the sequence length: the worst legal
configuration can be 4–5× slower than the best. Each kernel is therefore *tuned* on your device. The first device call
that finds no variant tunes the default lengths itself and says so, but it is better done deliberately, ahead of
latency-sensitive work:

```julia
tune_flash_kernel!(; dim=32)                    # :default,   lengths from tune_lengths(): 256 and 512
tune_flash_mma!(; dim=32, T=Float16)            # :flash_mma, Float16 storage
tune_flash_kernel!([768, 1024]; dim=32)         # also tune for larger proteins
set_tune_lengths!([256, 512, 768, 1024])        # make that the project default (LocalPreferences.toml)
```

What tuning does, for each length `n`:

1. builds the admissible configurations of each stage (group size, key tile, softmax sub-block for `:default`;
   simdgroups, row blocks and staged tile for `:flash_mma`), searching the grid exhaustively up to 32 points and
   by coordinate descent beyond;
2. **checks every candidate before its time counts**: it must be admitted by its compiled pipeline and reproduce a
   plain-loop reference on a small problem. Over-dispatching a pipeline returns garbage *and* an impossibly fast
   time, which would otherwise read as a huge win;
3. times the survivors on a triangle problem of exactly length `n`, and keeps the fastest per stage;
4. compiles every stage at that configuration for each combination of bias and mask being present (an absent
   operand is a binary of its own), and registers them as variants keyed
   `(dtype, dim, n, hasbias, hasmask)`.

A call of any length runs the variant tuned for the *closest* length. The attention geometry (outer extent, number
of heads) is deliberately not part of the key: measured on an M1 Pro over triangle, MSA-row and plain-sequence
shapes, one choice per device costs at most about 10 % against tuning every geometry.

The calibration (the measured times) is persisted next to the binaries, so repeating a tuning call is a file read:

```julia
calibrate!(; dim=32, n=512)          # measure :default at one length (also done by tune_flash_kernel!)
calibrate_mma!(; dim=32, n=512)      # the same for :flash_mma
clear_calibration!(; disk=true)      # forget measurements, in memory and on disk, e.g. after editing a kernel
```

Everything lives under `~/.cache/KernelDispatch/ops/LuxTriangleAttention.sdpa/<kernel>/`. Set
`KERNELDISPATCH_CACHE` to move the root (useful for tests and benchmarks that want an isolated cache). Variants built from other kernel sources are ignored and re-tuned on demand,
so updating the package never serves a stale binary.

## Precision

Storage is `Float32`, `Float16` or `BFloat16`; every accumulator, the softmax state, `lse` and `delta` are `Float32`
in both kernels. `:flash_mma` additionally tunes the operand type of the second matrix product of each step: `Float32`
or, for 16-bit storage, the storage type (the FlashAttention-2 convention). On an M1, BFloat16 operands fail
verification (Apple's compiler cannot assemble a bfloat fragment from registers), so BFloat16 storage runs with
Float32 operands there. See [Half precision](4-half-precision.md) for the rest.

`k`, `v`, the bias and the incoming gradient are converted to `q`'s storage type, so mixed-precision operands run
rather than being refused. A bias is converted with saturation at `±floatmax(T)/2`: a mask folded into the bias as
`1e9·(mask - 1)` is finite in Float32 but `-Inf` in Float16, and would turn a row into `NaN`.

## Gradients

Both kernels have a `backward`, and the layer is differentiable through it:

- **Eager Enzyme** on device arrays uses KernelDispatch' one rule on `forward`, which runs `backward`. Enzyme cannot
  differentiate the ordinary Metal.jl code around it, so on Apple hardware training goes through Reactant.
- **Under Reactant** the forward is a custom call in the compiled program. A gradient traced *through* a kernel
  throws until Enzyme-JAX can differentiate a custom call
  ([EnzymeAD/Enzyme#2516](https://github.com/EnzymeAD/Enzyme/issues/2516)); use `use_kernel=false` for differentiated
  traced programs, and see [Compiling with Reactant](3-reactant.md).
