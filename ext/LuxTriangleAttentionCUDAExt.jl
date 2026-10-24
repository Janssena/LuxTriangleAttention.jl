module LuxTriangleAttentionCUDAExt

# What CUDA supplies to the built-in attention kernels: the device intrinsics KernelAbstractions
# cannot express (the `dbias` atomic) and two small hooks on KernelDispatch' CUDA tag (`to_device`,
# `device_caps`). Everything else — routing, launch, timing, `ka_compile` — is KernelDispatch' CUDA
# extension, and building, verifying and tuning the stages is backend-agnostic code in the package
# (`src/kernels/launch.jl`, `src/kernels/tune.jl`). The Metal counterpart is `LuxTriangleAttentionMetalExt.jl`.

using CUDA
using CUDA: CUDACore
using LuxTriangleAttention
import KernelDispatch

const LTA = LuxTriangleAttention

LTA.to_device(::KernelDispatch.CUDABackendTag, x::AbstractArray) = CuArray(x)

# `max_shmem` is the STATIC shared memory a block may use: KernelAbstractions' `@localmem` is static, and CUDA
# caps that at 48 KiB per block whatever the device's opt-in maximum is (dynamic shared memory is out of scope).
# The name carries the compute capability: it keys the persisted calibration, and two cards of one family
# with different capabilities tune differently.
const STATIC_SHARED_BYTES = 48 * 1024

function LTA.device_caps(::KernelDispatch.CUDABackendTag)
    dev = CUDACore.device()
    cap = CUDACore.capability(dev)
    return LTA.DeviceCaps(32,
        Int(CUDACore.attribute(dev, CUDACore.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)),
        STATIC_SHARED_BYTES,
        "$(CUDACore.name(dev)) sm_$(cap.major)$(cap.minor)")
end

# The one thing `flash_bwd_dq!` needs that KernelAbstractions cannot express: an atomic add on the raw pointer
# `dbias` (see `_atomic_fadd!`'s docstring in `flash/backward.jl`). `idx` is 1-based, in elements; the byte offset
# is Int64, so no conversion here can throw (a kernel whose code can throw is refused at build time on CUDA).
LTA._atomic_fadd!(ptr::Core.LLVMPtr{Float32,CUDACore.AS.Global}, idx::Integer, val::Float32) =
    CUDACore.atomic_add!(ptr + (Int(idx) - 1) * sizeof(Float32), val)

end # module
