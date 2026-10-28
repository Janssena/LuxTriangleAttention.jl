# Changelog

All notable changes to LuxTriangleAttention.jl. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [0.5.0] — 2026-10-27

The half-precision release. `Float16` and `BFloat16` now run through the layers and the fused kernels without
overflowing, the attention-kernel machinery moved out to
[KernelDispatch.jl](https://github.com/Janssena/KernelDispatch.jl), and `TriMulCore` is laid out token-major. This
is a breaking release for anyone who registered their own kernels or relied on the old kernel cache
(see *Upgrading* below).

### Added

- **CUDA backend** (`ext/LuxTriangleAttentionCUDAExt`, loaded with `using CUDA`): the built-in `:default` kernel and the
  tensor-core `:flash_mma` kernel (`mma.sync` m16n8k16, Ampere sm_80 and newer) run on NVIDIA GPUs. The `:flash_mma`
  fragment hooks are now written against a fragment shape each backend states (`_mma_shape`; Metal 8×8×8, CUDA
  16×8×16 with two row slots per lane), so a backend is a handful of methods. On CUDA, `:flash_mma` takes Float16 and
  BFloat16 storage as it is; Float32 storage runs at Float16 (saturating cast in, results cast back to Float32), so
  Float32 data is rounded to 16 bits inside that kernel (`mma_compute_type`). `:default` keeps Float32 exact. A GPU
  older than sm_80 runs `:default`.
- The tuner registers, per stage, the fastest configuration that EVERY bias / mask specialisation can dispatch (the
  absent-operand binaries can admit fewer threads than the one that was measured).
- **16-bit data, end to end** (`src/precision.jl`; tutorial *Half precision* in the docs). Float16 tops out at
  65504, and two reductions over long axes overflow it on real protein activations. Both are now handled:
  - **`StableLayerNorm`** (exported): a `Lux.LayerNorm` wrapper that accumulates its statistics in Float32 for
    16-bit data (`layernorm_f32stats`: one pass, returned in the input's type). For any other input it IS the
    wrapped layer — same call, same traced program, bitwise the same result — so Lux's own dispatch on the
    array type is unchanged. Same parameter and state tree as `Lux.LayerNorm`, so checkpoints load either way;
    `ChannelsLast` supports it. `TriangleAttention`, `TriangleMultiplication` and `TriMulCore` use it, and
    `LayerNormNoBias` is now a factory for it (a `StableLayerNorm` around the frozen layer, same parameter and
    state tree as before). A Reactant method routes traced Float16/BFloat16 arrays to the Float32 path.
  - **Contraction modes for triangle multiplication.** The contraction `Σₖ aₖ·bₖ` overflows Float16 (AlphaFold2's
    multimer template stack reaches |X| ≈ 4.8e5 at N = 494). `TriMulCore` picks one of three paths statically
    from the data's element type and its state (`contraction_mode`):
    - `Folded(k)`: the value projections of `a` and `b` were pre-scaled by 2⁻ᵏ, so the contraction sums roughly
      a mean and stays in range in Float16. The output LayerNorm is scale-invariant and, with ε scaled by 16⁻ᵏ and
      Float32 statistics, removes the 4⁻ᵏ exactly. The normalisation runs in Float32 too, since the scaled ε
      makes 1/√(σ²+ε) about 1e6 for a token whose `X` is all zero (a fully masked pair): `Inf` in Float16, so
      that token would come out NaN. Fast, and valid in any precision.
    - `HalfSafe()`: 16-bit data with ordinary parameters (e.g. a plain `Lux.f16(ps)`): the contraction runs in
      Float32 and the output LayerNorm normalises in Float32 before narrowing. Always correct, slower and wider.
    - `FullPrecision()`: anything else, the ordinary path, unchanged.
  - **`fold_trimul(ps, st, k)`** prepares a layer's parameters for `Folded` (apply it to the Float32 parameters,
    before casting to 16 bits) and writes `fold_k = k` to the core's state. It scales only the value rows, whatever
    the projection layout (fused or separate `a`/`b` GLUs, each with a fused or its own gate Dense, with or without
    bias); **`fold_k(L)`** is the smallest `k` with
    `4ᵏ ≥ L`, a bound on the longest sequence the parameters must serve. The mode follows the *state*, so a layer
    can neither apply a 4ᵏ its weights do not carry nor miss one they do; an `Int` in a Lux state is a
    compile-time constant under Reactant, so the choice costs nothing at runtime. Folding is exact (powers of two
    do not change Float16 rounding, absent underflow): a different `k` gives the identical Float16 result, and
    folded Float32 equals unfolded Float32 to 1e-5. Against a Float64 reference on the same rounded inputs,
    `HalfSafe` is within 5e-3 and `Folded` within 1e-2 on a deliberately overflowing test, and `Folded` within
    6–7e-4 on AlphaFold2 activations on device.
  - **`mask_value(T)`**: the finite masking constant `-floatmax(T)/2`, to SELECT in where attention is masked
    (`ifelse.(keep, logits, mask_value(T))`) instead of `-Inf` or `-1e9`. The latter is `-Inf` in Float16 and
    turns a fully padded row into NaN. The rolled chunk path now masks with it.
  - Public (not exported): `ishalf`, `mask_value`, `fold_k`, `fold_trimul`, `contraction_mode`,
    `layernorm_f32stats`, `ContractionMode`, `FullPrecision`, `HalfSafe`, `Folded`.
- **BFloat16 storage** (`BFloat16s.BFloat16`, the type Metal.jl and Reactant share) for both attention kernels,
  eager and traced, next to Float32 and Float16: `supported_eltypes()`, `verification_tolerance(BFloat16) = 2e-2`.
- **`:flash_mma`, a matrix-unit attention kernel**, the second built-in kernel next to `:default` (which is
  unchanged and stays the default): `use_kernel!(:flash_mma)`, tuned with `tune_flash_mma!` / `calibrate_mma!`.
  It is one KernelAbstractions kernel that calls the backend's matrix-fragment intrinsics through a small set of
  hooks (implemented for Metal's simdgroup MMA in the extension; a CUDA/ROCm port implements the same hooks),
  so it is Metal only for now. On an M1 Pro, GPU kernel time against `:default`: forward 1.5–1.85× faster at
  Float32 and 1.6–1.7× at Float16; backward ~3.3–3.7× at Float32 and ~1.9–2.2× at Float16 (a training step ~3×
  and ~2×), with accuracy equal to `:default` within the storage type's rounding. Every accumulator stays Float32.
- **Tuned 16-bit operands in `:flash_mma`.** The operand type of each step's second matrix product (`P·V`;
  `P·dO`, `dS·Q`; `dS·K`) — Float32 or the 16-bit storage type, the FlashAttention-2 convention — is a tuning
  axis, verified like every parameter (also on a 256× scaled `dO`). Accuracy on an M1 Pro (Float16 storage) is
  unchanged within noise. On the M1, BFloat16 operands are rejected by verification (Apple's compiler cannot
  build a bfloat fragment from registers), so BFloat16 storage runs with Float32 operands there.
- **Tuning per sequence length.** Each length is calibrated and tuned AT that length, and a call runs the variant
  tuned for the closest length (ties to the larger). `tune_flash_kernel!(lengths; dim)` and
  `tune_flash_mma!(lengths; dim)` take one length or several; without lengths, and on the first device call with
  no variant, they tune `tune_lengths()` — default `[256, 512]`, persisted per project with `set_tune_lengths!`
  (e.g. `[256, 512, 768, 1024]` for larger proteins; new `Preferences` dependency). `calibrate!` and
  `calibrate_mma!` take `n`.
- **`ChannelsLast(layer)`** (exported): applies `Dense`, `LayerNorm`, `StableLayerNorm` or
  `GatedLinearUnit`/`SwiGLU` to channels-last data (`[..., C]`) with the wrapped layer's own parameter tree, so
  token-first operations hit faster matmul orientations and need no `permutedims`.
- **Documentation site** (`docs/`, Documenter.jl, deployed by `.github/workflows/docs.yml`): tutorials for the
  layers, the fused kernels and their tuning, Reactant and `chunk_size`, half precision and bringing your own
  kernel; a kernel-operand reference, the API reference and an internals page. README badges.

### Changed

- **Attention kernels are KernelDispatch.jl ops** (KernelDispatch 0.2, a registered package). Attention is the op
  `SDPA`; its kernels are `FlashScalar` (`:default`), `FlashMMA` (`:flash_mma`) and any you add. Select with
  `use_kernel!(SDPA, name)`, `use_kernel!(:sdpa, name)` or `use_kernel!(name)`. Selection, variants, binding and
  launching binaries, custom-call embedding and the Enzyme rule are KernelDispatch's, and:
  - a custom kernel is a type plus `forward`/`backward` methods registered with
    `KernelDispatch.register_kernel!(SDPA(), kernel)`, binding its binaries' arguments explicitly in Julia
    (`call_binary`); `docs/src/tutorials/5-custom-kernels.md`, with trifast as the example;
  - KernelDispatch binaries record the types they were compiled for and every launch is checked against them, so
    a launch binding another storage type or a 64-bit size is refused naming the argument. The built-in binaries
    carry their `signature`, and `_sdpa_prep` converts `k`, `v` (and the backward's `dO`) to `q`'s storage type,
    as it already did the bias, so mixed-precision operands run instead of being refused;
  - `stage_args(kernel, Val(stage), roles, scalars)` (`src/kernels/stage_args.jl`) builds every launch of a
    built-in binary — the op's `forward`/`backward` and the tuner's candidates alike — so what is verified and
    timed is bound exactly as what is called;
  - the builtins are compiled with `KernelDispatch.ka_compile`; launches pass `extent=(Nq, Nj, H·B)` and each
    binary records its queries per group as its `tile`;
  - the backward is no longer attached to the traced forward as `enzyme.reverse` (Enzyme-JAX ignores it), and
    Enzyme is no longer a (weak) dependency of this package;
  - gradients come back in their argument's element type (`dbias` for a Float16 bias).
- **The device kernels are ordinary source**, not an extension: KernelAbstractions is a regular dependency.
- **One tuning skeleton for both kernels** (`src/kernels/tune.jl`): a kernel declares only its stages, its
  parameter grid per stage, what is admissible without compiling, and how a point becomes a binary; search
  (full sweep up to 32 points, coordinate descent beyond), the build → pipeline-limit → verification →
  device-timing gate, calibration and registration are shared. Configurations are NamedTuples
  (`(; tg, kt, sb)`, `(; nsg, r, tile)`). Both kernels calibrate at one probe (4 heads). The public names
  (`tune_flash_kernel!`, `tune_flash_mma!`, `calibrate!`, `calibrate_mma!`, `flash_config`, `flash_mma_config`,
  `clear_calibration!`) are unchanged. Kernel tuning also takes `Nj` into account.
- **Backend-agnostic tuner, slim Metal extension.** Building, launching, verifying and timing stages live in
  `src/kernels/launch.jl` / `tune.jl` on KernelDispatch's public API. The Metal extension only adds the device
  intrinsics (`_atomic_fadd!`, `:flash_mma`'s fragment hooks) and two methods on KernelDispatch's Metal tag
  (`to_device`, `device_caps`).
- **Only tuned winners are kept on disk.** The tuner's candidate binaries (~100 per kernel, dtype and head dim)
  are compiled into a per-session scratch directory, deleted at exit; the winners are copied into the registry.
- **`TriMulCore` is token-major.** The projections are computed `xᵀ·Wᵀ → [N²·B, 2H]`, which IS the batched
  contraction's `[N, N]` planes layout, so the contraction runs on views with a lazy `batched_transpose` and no
  `permutedims`; the output LayerNorm and `glu_out` stay token-major, leaving one transpose of the final result.
  Same parameter tree (the sublayers are `ChannelsLast` wrappers), so weights load unchanged and the
  AlphaFold2/3/Boltz2 parity tests apply as before. Forward outputs are bitwise identical to the previous layout
  on CPU, eager Metal and Reactant; gradients agree to ≤ 5e-7. On an M1 Pro (fp32, C = H = 128): 1.65–2.2×
  faster forward and up to 1.29× faster Enzyme gradient under Reactant, 1.95–2.43× faster forward eagerly.
- **`TriMulCore` masks** may be `[N, N]` (shared across the batch) as well as `[N, N, B]`, and are applied out of
  place (fused into the GLU broadcast under Reactant), so they are differentiable. The old in-place masking of
  views is gone.
- **Flash kernels** take an optional mask and bias, and are faster overall.
- **A fully masked key set attends uniformly instead of returning NaN**, matching OpenFold and Boltz2 (finite
  `-1e9` mask bias): `o = mean(v)` over the keys, `dV = dO/n`, no gradient to `q`/`k`/`bias`. Applies to the
  default path, the rolled chunk path and both kernels, whose `lse = -Inf` marks the row for their backward. The
  tuner's verification problem includes one fully masked slice.
- **A bias converted to the storage type saturates** at `±floatmax(T)/2` instead of overflowing: a mask folded
  into a Float32 bias as `1e9·(mask - 1)` beside Float16 activations was `-Inf`, and a query whose every key it
  masked gave NaN. A query masked on every key through the bias alone is now finite.
- **The rolled Reactant chunk path is Float16-safe**: a Float16 chunk runs in Float32 (the compiler may move the
  `1/√d` scale past `Q·Kᵀ`, overflowing logits past 65504; one chunk's scores are all it holds). The plain
  `chunk_size=nothing` / `use_kernel=false` path is LuxLib's attention, not Float16-safe under Reactant, and is
  documented as not for production. The chunked CPU path needed no change.
- **Kernel cache.** The registry follows KernelDispatch's convention,
  `~/.cache/KernelDispatch/ops/LuxTriangleAttention.sdpa/<kernel>/`, with each kernel's calibrations beside it
  (`<kernel>/calibration/`, one file per device, head dim, dtype and length). `KERNELDISPATCH_CACHE` moves the
  root. Variants are keyed `(dtype, dim, N, hasbias, hasmask)` with the tuned `N` (format `k6`).

### Fixed

- **Fused kernels compiled with Reactant on NVIDIA GPUs returned wrong values** whenever XLA computed an operand before the
  call (the transposed pair bias, or q, k, v behind the projections): KernelDispatch's `stablehlo.custom_call` declared no
  operand layouts, so XLA's GPU compiler could pass the buffer in another memory order. KernelDispatch now declares the
  default dense layout for every operand and result (KernelDispatch 0.2.1, now the minimum; eager calls
  and Metal were never affected). The traced kernel tests in `test/ext/reactant/` cover it.
- **Reactant on NVIDIA GPUs multiplies Float32 matrices at TF32 precision by default** (a compiler default, about 3e-4
  relative error); the Reactant tests run at `PrecisionConfig.HIGHEST` on CUDA and the tutorial says how to compile
  with full Float32 products.
- The tuner picked each stage's configuration with the bias-and-mask build only, and could register one that a leaner
  specialisation could not dispatch (a thread limit of 448 against 512 measured); it now picks the fastest configuration
  every specialisation admits.
- `build!` rebuilt the storage type from `nameof(T)` with `getfield(Base, …)`, which has no `BFloat16` on Julia 1.13.
- `Lux.LayerNorm` accumulated the variance in Float16, so pair activations with σ ≈ 60 over 32 channels came
  out finite but 85–99% wrong. Compiled Float16 `TriangleAttention` and `TriangleMultiplication` now agree
  with Float64 to ~1e-3.
- Masks written as `1e9·(mask - 1)` were `-Inf` in Float16 and turned a fully padded window into NaN
  (see `mask_value` and the saturating bias above).
- The traced (Reactant) backward returned a wrong `dbias`.
- Eager Metal `use_kernel=true` silently fell back to LuxLib attention for fused q/k/v layers, because
  `SubArray`s of an `MtlArray` were classified as CPU arrays.

### Removed

- `_sdpa_kernel` / `_sdpa_kernel_bwd!` (`src/kernels/backend.jl`): `_sdpa` calls `SDPA()(q, k, v, mask, bias)`
  directly, and the backward is `KernelDispatch.backward`.
- This package's own kernel registry: `register_kernel!(; fwd=…, stages=…)`, `unregister_kernel!`, the
  `LTA_KERNEL` environment variable, `kernel_cache_root` and the `LTA_KERNEL_CACHE` override (use
  `KERNELDISPATCH_CACHE`).
- The Enzyme extension (the rule lives in KernelDispatch).
- The tuner's `FlashConfig`, `candidate_configs`, `search_configs` and `nbucket` (sequence lengths are tuned
  exactly, not bucketed).

### Upgrading

- Kernels tuned under 0.4.0 are ignored and re-tuned on demand, at the lengths in `tune_lengths()`. Calibrations
  are re-measured once. `~/.cache/LuxTriangleAttention` is unused and can be deleted.
- Float16/BFloat16 models work as-is (`HalfSafe`). For the fast path, `fold_trimul` the Float32 parameters of
  each `TriangleMultiplication` before casting with `Lux.f16`:
  `ps, st = fold_trimul(ps, st, fold_k(L))` for the longest sequence `L` the model will see.
- Custom kernels need porting to KernelDispatch: see the *Your own attention kernel* tutorial.

## [0.4.0] — 2026-09-16

The kernel release.

- New chunking and kernel-selection setup (`src/layers/chunk.jl`).
- Kernel registry and device abstraction (`src/kernels/registry.jl`, `device.jl`, `backend.jl`);
  hashes in the registry so several kernels can coexist.
- KernelAbstractions flash-attention kernel.
- Kernel tuning, including `nouter` support for tuning `Ni` and `Nj` simultaneously.
- Kernels are guarded against cross-attention; `prep_bias` dispatches on tuples.
- Metal extension.
- Enzyme extension with memory-efficient backward pass on GPU.
- Reactant extension.
- Tests for chunking, attention-layer routing and kernels; dependencies updated.

## [0.3.6]

- Simplified the extensions and removed `map` over varargs.
- Cleaned up the GLU and triangle-attention implementations.
- Started work on chunking and kernel selection (landed in 0.4.0).

## [0.3.5]

- Reactant extension now uses traced arrays.

## [0.3.4]

- Fix: support Reactant mask types in the extension.
- Benchmarks for the `Attention` layer, benchmark utilities, README benchmark section.

## [0.3.3]

- Fix: `gate_maybe` bug.

## [0.3.2]

- Rank selection for triangle multiplication.
- Docs.

## [0.3.1]

- Triangle attention and triangle multiplication can be overloaded (extended) by downstream packages.

## [0.3.0] — 2026-05-08

- Bias shapes are handled internally for both triangle attention and pair-bias attention
  (`prep_bias`, `bias_layout=:qk` / `:kq`).
- README rewrite.

## [0.2.0] — 2026-04-28

- Switched attention to `Lux.scaled_dot_product_attention`; removed the hand-written kernels
  and the extension that carried them.
- Docs and README updated.

## [0.1.0] — 2026-04-16 to 2026-04-22

Initial development.

- Julia ↔ Python parity implementations, plus Tullio and AMX-optimised variants and a generic
  GPU implementation.
- `TriangleAttention` and `Attention` layers with Enzyme rules; forward and backward parity with
  AlphaFold2, AlphaFold3 and Boltz2 references.
- `TriangleMultiplication` with AlphaFold2 parity.
- Uses LuxLib `batched_matmul` instead of NNlib; removed Functors and NNlib dependencies.
- GLU primitives removed; AD-backend-agnostic backward tests.
- License added, compat updated.
