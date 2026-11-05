module LuxTriangleAttentionCUDAExt

# What CUDA supplies to the built-in attention kernels: the device intrinsics KernelAbstractions
# cannot express (the `dbias` atomic, `:flash_mma`'s fragment hooks). Everything else — routing, launch, timing,
# `ka_compile`, and the device tools `to_device`, `device_caps`, `release!` — is KernelDispatch' CUDA
# extension, and building, verifying and tuning the stages is backend-agnostic code in the package
# (`src/kernels/launch.jl`, `src/kernels/tune.jl`). The Metal counterpart is `LuxTriangleAttentionMetalExt.jl`.

using CUDA
using CUDA: CUDACore
using LuxTriangleAttention
import KernelDispatch

const LTA = LuxTriangleAttention

# The one thing `flash_bwd_dq!` needs that KernelAbstractions cannot express: an atomic add on the raw pointer
# `dbias` (see `_atomic_fadd!`'s docstring in `flash/backward.jl`). `idx` is 1-based, in elements; the byte offset
# is Int64, so no conversion here can throw (a kernel whose code can throw is refused at build time on CUDA).
LTA._atomic_fadd!(ptr::Core.LLVMPtr{Float32,CUDACore.AS.Global}, idx::Integer, val::Float32) =
    CUDACore.atomic_add!(ptr + (Int(idx) - 1) * sizeof(Float32), val)

# The matrix-unit kernel `:flash_mma`'s fragment hooks.
include("LuxTriangleAttentionCUDAExt/flash_mma.jl")

end # module
