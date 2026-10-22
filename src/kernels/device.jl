# What the built-in kernels need from a device backend, as methods on KernelDispatch' backend tags. A
# backend extension (Metal: `ext/LuxTriangleAttentionMetalExt.jl`) adds them for its own tag;
# everything else — building, launching, verifying, timing, tuning — is backend-agnostic code in
# `launch.jl` and `tune.jl`, on KernelDispatch' public API.

"""
    device_kernel_available() -> Bool

Whether a backend extension is loaded AND its device is usable (`KernelDispatch.device_available()`).
False on a host-only setup.
"""
device_kernel_available() = KernelDispatch.device_available()

"""
    DeviceCaps

What a backend reports for the candidate configurations to be built. Deliberately small: anything
that can be measured is measured rather than declared.

# Fields
- `warp`: threads per warp / wavefront / SIMD group. Group sizes are multiples of this.
- `max_threads`: the device's ceiling on threads per group. A NECESSARY but not sufficient bound —
  a compiled pipeline's own limit (`KernelDispatch.max_threads`) is checked per candidate.
- `max_shmem`: bytes of local memory available to one group.
- `name`: device identifier, used as the calibration key.
"""
struct DeviceCaps
    warp::Int
    max_threads::Int
    max_shmem::Int
    name::String
end

"""
    device_caps(backend) -> DeviceCaps

The device's limits. Implemented by the backend extension for its own tag.
"""
device_caps(be::KernelDispatch.KernelBackend) = throw(ArgumentError(
    "no device capabilities for $(typeof(be)); load the package for that backend"))

"""
    to_device(backend, x::Array) -> device array

`x` copied to the backend's device. Implemented by the backend extension; the tuner's probe and
verification problems are built on the host and moved with it.
"""
to_device(be::KernelDispatch.KernelBackend, x) = throw(ArgumentError(
    "no device arrays for $(typeof(be)); load the package for that backend"))

"""
    supported_eltypes() -> Tuple{Vararg{DataType}}

Storage element types the built-in kernels will run.

The kernels are written generically over the storage type (accumulation is always `Float32` — see
note 3 in `src/kernels/flash/forward.jl`). A dtype is listed only after the reference comparison has
run at it: a path that has only ever been executed at `Float32` is untested, not working. `Float16` and
`BFloat16` (BFloat16s.jl's, the type Metal.jl and Reactant use) are checked against the reference on
upcast inputs, on the eager and traced paths (`test/ext/metal/launch.test.jl`,
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
