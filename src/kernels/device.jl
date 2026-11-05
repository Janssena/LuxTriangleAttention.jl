# What the built-in kernels need from a device backend that is specific to them, as methods on KernelDispatch'
# backend tags. A backend extension (Metal: `ext/LuxTriangleAttentionMetalExt.jl`) adds them for its own tag. The
# generic device tools — `to_device`, `device_caps`, `release!`, `device_available` — are KernelDispatch's; everything
# else — building, launching, verifying, timing, tuning — is backend-agnostic code in `launch.jl` and `tune.jl`.

"""
    supported_eltypes() -> Tuple{Vararg{DataType}}

Storage element types the built-in kernels will run.

The kernels are written generically over the storage type (accumulation is always `Float32` — see
note 3 in `src/kernels/flash/forward.jl`). A dtype is listed only after the reference comparison has
run at it: a path that has only ever been executed at `Float32` is untested, not working. `Float16` and
`BFloat16` (BFloat16s.jl's, the type Metal.jl and Reactant use) are checked against the reference on
upcast inputs, on the eager and traced paths (`test/kernels/launch.gpu.test.jl`,
`test/ext/reactant/attention_kernel.test.jl`). They are the precisions KernelDispatch 0.2 supports
(eagerly, from a trace, and in launches that mix them), and its signature check refuses a launch whose
buffers are not of the type the binary was compiled for.
"""
supported_eltypes() = (Float32, Float16, BFloat16)

function _check_eltype(::Type{T}) where {T}
    T in supported_eltypes() && return nothing
    throw(ArgumentError(
        "storage eltype $T is not enabled for the device attention kernels (enabled: " *
        "$(join(supported_eltypes(), ", "))). The kernels are generic over storage precision and " *
        "$T may well compile, but it has not been validated against the reference — see " *
        "`supported_eltypes`."))
end

"""
    kernel_source_tag() -> String

Short hash of the kernel sources, for a backend to put in its binary cache key.

The cache key that matters — the stage, `T`, `dim` and the tuned parameters — says what a binary
was compiled FOR, not what it was compiled FROM. Without this, editing `flash/forward.jl` leaves every
previously built binary on disk under a key that still matches, so the next run silently launches
the OLD kernel: no error, no recompile, and results that quietly belong to code that no longer exists.

Computed once, when the package is precompiled, over the kernel source files — so it tracks the code
actually baked into the loaded image. It does NOT cover the toolchain (Julia, GPUCompiler,
KernelAbstractions, or the backend's own compiler), which can also change codegen.
"""
kernel_source_tag() = KERNEL_SOURCE_TAG

const KERNEL_SOURCE_TAG = let h = UInt64(0)
    for f in ("forward.jl", "backward.jl")
        h = hash(read(joinpath(@__DIR__, "flash", f), String), h)
    end
    string(h % UInt32; base=36)
end

"""
    trimul_tunables(backend) -> NamedTuple

The tile and block sizes of `TriMul`'s `:default` kernel on this backend:

- `TB`: tokens per LayerNorm group, shrunk automatically until the group's local memory fits the device;
- `NT`: threads per LayerNorm group (a multiple of `TB`);
- `NTE`: threads per group of the elementwise stages;
- `nb`: tokens per block of the input and output stages. It bounds the `[nb, 4H]` projection and the other
  per-block workspaces: larger blocks mean fewer, larger GEMMs and more memory.

The default is the optimum measured on an Apple M1 Pro (docs/FEATURE_trimul_kernel.md). A backend extension
overloads it for its tag.
"""
trimul_tunables(::KernelDispatch.KernelBackend) = (TB=32, NT=256, NTE=256, nb=16384)

"""
    trimul_contraction_type(backend, T, mode) -> Type

The storage type of `TriMul`'s `a`, `b` and contraction result `X` for data of type `T` in contraction `mode`
([`ContractionMode`](@ref)): the type the batched matrix product `X = a ⊗ b` runs in. The default is `Float32` on the
[`HalfSafe`](@ref) path, whose contraction overflows 16 bits, and `T` otherwise (a [`Folded`](@ref) contraction is in
range, and in 16 bits it halves the memory and traffic of `a`, `b` and `X`). A backend whose 16-bit batched product is
slow overloads it to say `Float32`: Metal.jl's runs 7× slower than its Float32 one at N = 512 (88 ms against 13 ms),
so `LuxTriangleAttentionMetalExt` widens there. The result is the same or better (the Float32 product is the more
accurate one), only wider.
"""
trimul_contraction_type(::KernelDispatch.KernelBackend, ::Type{T}, mode) where {T} = mode isa HalfSafe ? Float32 : T

const TRIMUL_TUNABLE_OVERRIDE = Ref{NamedTuple}(NamedTuple())

"""
    set_trimul_tunables!(; TB, NT, NTE, nb)
    set_trimul_tunables!()

Override some of [`trimul_tunables`](@ref) for this process (the others keep the backend's values), or with no arguments
clear the override. For tuning on a new device (`test/bench/kernels/trimul_tune.bench.jl`): each combination of `TB`,
`NT` and `NTE` is a variant of its own, compiled on first use, while `nb` is only a host-side block size. A value the
device cannot run (a LayerNorm tile that does not fit its local memory) is shrunk, as for the default.
"""
function set_trimul_tunables!(; kw...)
    bad = setdiff(keys(kw), (:TB, :NT, :NTE, :nb))
    isempty(bad) || throw(ArgumentError("unknown TriMul tunable(s) $(collect(bad)); they are TB, NT, NTE, nb"))
    TRIMUL_TUNABLE_OVERRIDE[] = NamedTuple(kw)
    return nothing
end

# The tunables in force: the backend's, with the process's override on top.
_trimul_tunables(be) = merge(trimul_tunables(be), TRIMUL_TUNABLE_OVERRIDE[])
