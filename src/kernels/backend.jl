"""
    KernelBackend

Where an attention call will execute. Not the array type: a `TracedRArray` is the same Julia type
whether it will run on CPU or GPU, and the distinction only exists in the XLA client.
"""
abstract type KernelBackend end

"""Host execution: plain Julia arrays, or tracing against a CPU client."""
struct CPUBackend <: KernelBackend end

"""Apple GPUs, via `MtlArray` or an `mps` client."""
struct MetalBackendTag <: KernelBackend end

"""NVIDIA GPUs, via `CuArray` or a CUDA client."""
struct CUDABackendTag <: KernelBackend end

"""AMD GPUs, via `ROCArray` or a ROCm client."""
struct ROCmBackendTag <: KernelBackend end

"""A device we have no kernel for. Always falls back rather than failing."""
struct UnknownBackend <: KernelBackend end

const GPUBackend = Union{MetalBackendTag,CUDABackendTag,ROCmBackendTag}

"""
    backend_of(x) -> KernelBackend

The backend an array will execute on.

Plain `AbstractArray`s are host memory. Backend extensions add methods for their own array types
(`MtlArray`, `CuArray`, `ROCArray`), and the Reactant extension adds one for traced arrays that
inspects the compiling client — which is the only way to tell a CPU trace from a GPU trace, since
the Julia type is identical.
"""
backend_of(::AbstractArray) = CPUBackend()
backend_of(x, xs...) = backend_of(x)

"""
    kernel_cache_root() -> String

Root of the on-disk kernel cache, overridable with `LTA_KERNEL_CACHE`.

Three things live under it, kept in separate subtrees because each is keyed differently:

  * `bin/<backend>/` — RAW compiled kernel binaries, as a backend extension's `build`/`build_bwd`
    produce them, keyed by `(T, dim, tg, kt)`. **Not** keyed by device: a Metal `.metallib` is AIR
    and is portable across Apple GPUs, and the same is true of PTX across a CUDA architecture.
    Recompiling per machine would be waste.
  * `calibration/` — measured configuration timings, keyed by `(device, dim, T)`. These ARE
    device-specific even though the binaries are not: the optimum is an occupancy property.
  * `kernels/<name>/` — the REGISTRY: [`register_kernel!`](@ref) (and [`tune_flash_kernel!`](@ref),
    for the reserved `:$(RESERVED_KERNEL_NAME)` entry) copy binaries here under stable names, one
    subdirectory per registered kernel, alongside a TOML manifest. This is what
    [`load_registry!`](@ref) rehydrates on the next Julia start — see `src/kernels/registry.jl`.
    Distinct from `bin/`: `bin/` is this package's own build cache, keyed purely on compile
    configuration; `kernels/` is keyed on the NAME the user selects with `use_kernel!`, and a single
    compiled binary in `bin/` can be copied into more than one registry entry.

Collapsing these would force either needless recompilation, reuse of one GPU's tuning on a
different one, or a registry entry that cannot survive its source binary moving.
"""
kernel_cache_root() = get(ENV, "LTA_KERNEL_CACHE",
    joinpath(homedir(), ".cache", "LuxTriangleAttention"))

binary_cache_dir(backend::AbstractString) =
    joinpath(kernel_cache_root(), "bin", lowercase(backend))

calibration_cache_dir() = joinpath(kernel_cache_root(), "calibration")

"""
    _sdpa_kernel(q, k, v, mask, bias) -> (output, lse)

Route an attention call to the selected kernel, or to the host path when there is no device.

Dispatches on [`backend_of`](@ref) rather than on the array type, so a traced array and a concrete
one reach the same decision from the same trait — see the note at the top of this file.

Always returns a PAIR. `lse` is the per-query log-sum-exp on a device path — the residual a
hand-written reverse rule needs to recompute attention weights a tile at a time, instead of caching
the full `Nk×Nq×H` weight tensor a flash-style forward never materialises — and `nothing` on the
host path, where Enzyme differentiates the ordinary softmax directly and there is no residual to
carry. Returning the pair unconditionally is what lets `_sdpa` unwrap with a plain `first`, rather
than testing the shape of the result at runtime.

# Host arrays never fail
On [`CPUBackend`](@ref) this runs [`_sdpa_default`](@ref) regardless of which kernel is selected.
That is deliberate: a user who has selected their own CUDA kernel and then passes host arrays has
not put their data on the GPU, and running the model's forward on CPU is a thing to support, not to
refuse. The selection is not silently *lost* — it applies the moment the arrays are on the device.
"""
_sdpa_kernel(q, k, v, mask, bias) = _sdpa_kernel(backend_of(q), q, k, v, mask, bias)

_sdpa_kernel(::CPUBackend, q, k, v, mask, bias) = (_sdpa_default(q, k, v, mask, bias), nothing)

function _sdpa_kernel(::UnknownBackend, q, k, v, mask, bias)
    @warn """
    Unrecognised device backend; falling back to the host attention path.
    Register a kernel with `register_kernel!` if you have one for this device.
    """ maxlog = 1
    return (_sdpa_default(q, k, v, mask, bias), nothing)
end

"""
    _sdpa_kernel(backend::GPUBackend, q, k, v, mask, bias) -> (output, lse)

The device path: resolve the selected entry's compiled stages for this problem and launch the
forward.

This method — not the 5-argument one — is what `LuxTriangleAttentionEnzymeExt` attaches its reverse
rule to, narrowed on `Const{<:GPUBackend}`. Enzyme rules dispatch on types only, so the CPU/GPU
split has to be visible to dispatch for the rule to be prevented from firing on host arrays; the
backend tag is what makes it so.

Unlike the host path there is **no fallback here**. A GPU array with no usable kernel is a
configuration error, and silently running host attention on it would hide a 10-100x slowdown behind
a correct answer.
"""
function _sdpa_kernel(be::GPUBackend, q, k, v, mask, bias)
    q, k, v, mask, bias = map(x -> device_dense(be, x), (q, k, v, mask, bias))
    entry = kernel_entry()
    entry.backend === backend_symbol(be) || throw(ArgumentError(
        "the selected kernel `:$(entry.name)` is a `:$(entry.backend)` kernel, but these arrays " *
        "are on `:$(backend_symbol(be))`. Call `use_kernel!` with a kernel for this device."))
    T = _realtype(eltype(q))
    dim, n = size(q, 1), size(q, 3)
    # The call's geometry, for per-layer tuned variants: `[hd, H, Nq, B]` is a plain sequence
    # (outer extent 1), `[hd, H, Nq, Nj, B]` has `Nj` outer slices; `heads` counts every
    # `(head, batch)` slice.
    nouter = ndims(q) == 5 ? size(q, 4) : 1
    heads = size(q, 2) * size(q, ndims(q))
    stages = resolve_variant(entry, T, dim, n; nouter, heads)
    stages === nothing && (stages = _ensure_variant(entry, T, dim, n; nouter, heads))
    stages = specialise_stages(entry, stages, T, dim, bias !== nothing, mask !== nothing)
    ndims(q) == 4 || return _sdpa_kernel_launch(be, entry, stages, q, k, v, mask, bias)
    o, lse = _sdpa_kernel_launch(be, entry, stages, _seq5(q), _seq5(k), _seq5(v), mask, bias)
    return reshape(o, size(q)), reshape(lse, 1, size(q, 3), size(q, 2), size(q, 4))
end

"""
    _seq5(x)

A plain sequence's `[hd, H, N, B]` operand as the rank-5 `[hd, H, N, 1, B]` every launcher speaks:
attention over `N` batched over an outer extent of ONE. A pure reshape — the memory order is
identical — and the mask (`[N, 1, 1, B]` from `prep_mask`) and bias (`[Nk, Nq, H, B]`) need none,
since the launchers read them by element count as `[N, 1, B]` and `[N, N, H, B]`.
"""
function _seq5(x::AbstractArray{T,4}) where T 
    hd, H, N, B = size(x)
    return reshape(x, hd, H, N, 1, B)
end

# The backward for a plain sequence: the rank-5 backward on reshaped operands, gradients reshaped
# back. `dbias` is returned as the launcher produced it; callers reshape it to `size(bias)`.
function _sdpa_kernel_bwd!(q::AbstractArray{T,4}, k, v, mask, bias, o, lse, dO) where T
    dq, dk, dv, dbias = _sdpa_kernel_bwd!(_seq5(q), _seq5(k), _seq5(v), mask, bias,
        _seq5(o), reshape(lse, size(lse, 1), size(lse, 2), size(lse, 3), 1, size(lse, 4)), # TODO: can lse not use _seq5?
        _seq5(dO))
    return reshape(dq, size(q)), reshape(dk, size(k)), reshape(dv, size(v)), dbias
end

"""
    backend_symbol(backend) -> Symbol

The registry's name for a backend tag, so an entry's `backend` field can be compared against the
tag `backend_of` returned.
"""
backend_symbol(::MetalBackendTag) = :metal
backend_symbol(::CUDABackendTag) = :cuda
backend_symbol(::ROCmBackendTag) = :rocm
backend_symbol(::KernelBackend) = :none

"""
    _ensure_variant(entry, T, dim, n) -> Vector{KernelStage}

Obtain compiled stages for a configuration the entry does not yet hold.

For `:$(RESERVED_KERNEL_NAME)` this tunes and compiles on the spot — the kernel is ours, we know how
to build it, and a fresh install should just work. It costs seconds, once, and says so.

For any other entry there is nothing to build: those binaries came from the user, so the only
correct response is to say which configuration is missing.
"""
function _ensure_variant(entry::KernelEntry, ::Type{T}, dim::Integer, n::Integer;
    nouter::Integer=n, heads::Integer=0) where {T}
    if entry.name === RESERVED_KERNEL_NAME
        _tracing() && throw(ErrorException(
            "no compiled `:$RESERVED_KERNEL_NAME` kernel for (T=$T, dim=$dim, n=$n), and it " *
            "cannot be built during tracing — compiling it runs eager device work, which the " *
            "tracer would capture. Run `tune_flash_kernel!($n; dim=$dim, T=$T)` before compiling."))
        @info "Tuning the built-in flash kernel for n=$n, dim=$dim, T=$T. " *
              "This runs once and is cached; call `tune_flash_kernel!` ahead of time to avoid it here."
        tune_flash_kernel!(n; dim, T)
        stages = resolve_variant(kernel_entry(RESERVED_KERNEL_NAME), T, dim, n; nouter, heads)
        stages === nothing && error("tuning `:$RESERVED_KERNEL_NAME` produced no usable variant")
        return stages
    end
    throw(ArgumentError(
        "the selected kernel `:$(entry.name)` has no variant for (T=$T, dim=$dim, n=$n). " *
        "Registered variants: $(sort(collect(keys(entry.variants)); by=string)). " *
        "Compile that configuration and add it with `register_kernel!`."))
end

"""
    _tracing() -> Bool

Whether we are inside a Reactant trace. `false` in the base package; the Reactant extension
replaces the underlying predicate.

This exists so [`_ensure_variant`](@ref) can refuse to tune mid-trace. Tuning builds real device
arrays and times real launches — ordinary eager host work that the tracer would otherwise capture
into the graph, at which point the device array constructor rejects the traced element type.
"""
_tracing() = _TRACING[]()
const _TRACING = Ref{Function}(() -> false)

"""
    _sdpa_kernel_launch(backend, entry, stages, q, k, v, mask, bias) -> (output, lse)

Run an entry's `:fwd` stage. Implemented by the backend extensions — the concrete-array launcher in
the backend's own extension, the `stablehlo.custom_call` emitter in the Reactant one — which is why
they dispatch on the array type as well as the tag.
"""
function _sdpa_kernel_launch end
