# 3. Compiling with Reactant

```@meta
CurrentModule = LuxTriangleAttention
```

Loading [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) enables the package extension
`LuxTriangleAttentionReactantExt`, and every layer can then be compiled with `@compile` (or run with `@jit`) to an
XLA program, on the CPU or, through the jax-mps PJRT plugin, on an Apple GPU. The layers need no changes: they are written so the traced version is the same code.

```julia
using Lux, LuxTriangleAttention, Reactant, Random

dev = Reactant.to_rarray
attn = TriangleAttention(128, 32, 4)
ps, st = Lux.setup(Random.default_rng(), attn)
x = randn(Float32, 128, 64, 64, 1)

f(l, x, ps, st) = first(l(x, ps, st))
compiled = @compile f(attn, dev(x), dev(ps), dev(st))
y = Array(compiled(attn, dev(x), dev(ps), dev(st)))
```

Gradients go through Enzyme as usual (`Enzyme.gradient(Reverse, loss, ...)` inside the compiled function). The
layers' token-major layout, which makes the contractions plain batched matrix products, was chosen by measuring under
Reactant, where it is up to roughly 2× faster than the channel-first formulation.

## Bounding memory with `chunk_size`

`Attention(...; chunk_size=n)` runs the attention in slices of `n` along dimension 4 of the prepared `q`, `k`, `v`.
Attention runs over dimension 3, so dimension 4 is the first axis it batches over, and for a pair input
`[C, Ni, Nj, B]` that is **`Nj`**, a token axis, not the sample batch. The live score tensor shrinks from
`[Nq, Nk, H, Nj, B]` to `chunk_size` slices of it, and it does so at `B == 1`. (For a 3-D input `[C, N, B]`
dimension 4 *is* the batch, so `chunk_size` does nothing at `B == 1`.) This is `chunk_size_tri_attn` of OpenFold and
Boltz.

```julia
attn = TriangleAttention(128, 32, 4; chunk_size=16)
```

Eagerly on the CPU this is an ordinary loop and bounds memory as you would expect (3.2 GB less peak for about 12 % more
time on one Pairformer layer at N = 512). On a GPU it is the wrong tool, because a fused kernel never materialises the
scores in the first place: the layer warns once, and [`use_kernel`](2-fused-kernels.md) is what you want.

### Keeping the loop rolled

Inside `@compile` the Julia loop would be **unrolled** into the graph, and because the chunks are independent XLA is
free to run them concurrently and keep every slice live, which cancels the benefit. The extension therefore
replaces the loop with a rolled `@trace` loop, a `stablehlo.while`. Emitting the `while` is still not enough: two
Enzyme-HLO passes can turn it back into the unchunked graph.

- `greedy_while_loop_batch_fission` sees the iterations are independent and fuses them into one batched
  `dot_general`. It fires at any chunk count.
- the while-loop unroller fires when `nchunks <= Reactant.Compiler.WHILE_UNROLL_THRESHOLD[]` (default 4).

[`chunked_compile_options`](@ref) holds off the first, and the threshold is a global you set yourself:

```julia
Reactant.Compiler.WHILE_UNROLL_THRESHOLD[] = 0
opts = LuxTriangleAttention.chunked_compile_options()
compiled = @compile compile_options=opts f(attn, dev(x), dev(ps), dev(st))
```

Check that it worked, rather than assuming, by looking at the optimised module:

```julia
mod = @code_hlo compile_options=opts f(attn, dev(x), dev(ps), dev(st))
LuxTriangleAttention.check_rolled(mod)       # true: a `stablehlo.while` is still there
```

The numbers are identical either way, which is how an unrolled, memory-hungry program goes unnoticed.
The threshold matters most for the **reverse** pass: Reactant's checkpointing nests the loop in two, with trip counts
of order `sqrt(nchunks)` that fall under the default threshold at essentially any chunk count, and the layer warns
once if the threshold is not `0`. On some Reactant versions an unrolled reverse does not even compile.

!!! warning "Verify the reverse before relying on a chunk size"
    Differentiating the rolled loop hits an unfixed Reactant/Enzyme defect for some configurations: the chunk write
    becomes a `dynamic_update_slice` with an update one rank too many (`update rank does not match operand rank`),
    which aborts the process. The forward is unaffected. No rule predicts it; on a bare `Attention` (chn = 32,
    head_dim = 8, 4 heads) it was the same at every `N` tried:

    | nchunks | 2 | 3 | 4 | 5 | 6 |
    |:--------|:-:|:-:|:-:|:-:|:-:|
    | reverse | ✗ | ✗ | ✓ | ✗ | ✓ |

    A given `nchunks` may work or fail depending on the surrounding layer. Try the gradient before committing to one.

Chunk sizes that do not divide the axis fall back to the unrolled form, with the same result.

## Fused kernels in a compiled program

With `use_kernel=true` and a GPU client the kernel is embedded as a `stablehlo.custom_call`: on Metal one that jax-mps runs,
on NVIDIA hardware one to XLA's kernel-call handler (Reactant's own client, no setup). On Metal this
needs the Metal PJRT client to be the default (Reactant's own default is the CPU client, under which a traced array is
a host array and the layer takes the host path):

```julia
using Reactant, Metal
Reactant.set_default_backend(client)         # the jax-mps client, see test/mps_client.jl (it finds the plugin via JAX_MPS_PLUGIN)
```

Three things to know:

- **Tune ahead of `@compile`.** Compiling a kernel is eager device work, which the tracer would capture, so a missing
  variant cannot be built mid-trace and the compile throws, naming the call to run. `tune_flash_kernel!` (or
  `tune_flash_mma!`) once, for the head dimension and storage type you compile with; any lengths will do, since a call
  runs the closest tuned one. See [Fused attention kernels](2-fused-kernels.md#Tuning).
- **KernelDispatch must declare layouts (CUDA).** XLA's GPU compiler may hand a kernel an operand it computed itself (the
  transposed pair bias, or q, k, v after the projections) in another memory order unless the custom call says every
  operand is dense in the default layout. KernelDispatch declares that (`operand_layouts` / `result_layouts`; see its
  *Traced launches* page), and a KernelDispatch without it silently returns wrong values for a fused kernel compiled
  with Reactant on an NVIDIA GPU: the symptom is a relative error of order 1 whenever a bias is present, while eager calls
  are exact. The traced kernel tests (`test/ext/reactant/`) check it on every run.
- **Forward only.** A gradient traced *through* a kernel throws until Enzyme-JAX can differentiate a custom call
  ([EnzymeAD/Enzyme#2516](https://github.com/EnzymeAD/Enzyme/issues/2516)). To train under Reactant use the
  unfused layer (optionally with `chunk_size`), and use the kernel for inference. You can still call the kernels'
  `backward` directly.

## Float32 on NVIDIA GPUs: TF32

XLA multiplies Float32 matrices at **TF32 precision** by default on NVIDIA GPUs from Ampere on (the default `dot_general`
precision): about 3e-4 relative error against Float32, measured on an A40 for compiled attention. This is the compiler's
choice, not this package's, and it applies to every matrix product in a compiled program (the projections and the plain
softmax path; the fused kernels do their own arithmetic). To compile with full Float32 products, set the precision for the
trace:

```julia
using Reactant
Reactant.with_config(dot_general_precision=Reactant.PrecisionConfig.HIGHEST,
                     convolution_precision=Reactant.PrecisionConfig.HIGHEST) do
    compiled = @compile model(x, ps, st)
    compiled(x, ps, st)
end
```

The package's own Reactant tests run this way on CUDA. Metal (jax-mps) and the CPU client multiply in Float32 already.

## Float16

A rolled chunk runs in Float32 inside Float16 programs. The compiler may move the `1/√d` scale past `Q·Kᵀ`, and the
unscaled logits reach 9e4 on Boltz2's activations, past `floatmax(Float16) = 65504`. The plain path (no chunking, no
kernel) is LuxLib's attention, which is not Float16-safe under Reactant and materialises every score: it is for
reference and tests. Use `chunk_size` or `use_kernel` for 16-bit data. More in [Half precision](4-half-precision.md).
