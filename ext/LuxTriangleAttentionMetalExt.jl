module LuxTriangleAttentionMetalExt

# What Metal supplies to the built-in attention kernels: the device intrinsics KernelAbstractions
# cannot express (the `dbias` atomic, `:flash_mma`'s fragment hooks) and two small hooks on
# KernelDispatch' Metal tag (`to_device`, `device_caps`). Everything else — routing, launch,
# timing, `ka_compile` — is KernelDispatch' Metal extension, and building, verifying and tuning the stages
# is backend-agnostic code in the package (`src/kernels/launch.jl`, `src/kernels/tune.jl`).

import Metal: AS

using Metal
using LuxTriangleAttention
import KernelDispatch

const LTA = LuxTriangleAttention

LTA.to_device(::KernelDispatch.MetalBackendTag, x::AbstractArray) = MtlArray(x)

LTA.device_caps(::KernelDispatch.MetalBackendTag) = LTA.DeviceCaps(32,
    Int(Metal.device().maxThreadsPerThreadgroup.width),
    Int(Metal.device().maxThreadgroupMemoryLength), String(Metal.device().name))

# The one thing `flash_bwd_dq!` needs that KernelAbstractions cannot express: an atomic add on the
# raw pointer `dbias` (see `_atomic_fadd!`'s docstring in `flash/backward.jl`). Metal has a native
# `atomic_fetch_add_explicit` for `Float32` device pointers (Metal 3.0+); the byte-offset arithmetic
# here is exactly what `pointer(A, idx)` does for any device array, so this is the same operation
# `@atomic A[idx] += val` performs on a real `MtlArray` — just spelled out for a raw pointer instead.
LTA._atomic_fadd!(ptr::Core.LLVMPtr{Float32,AS.Device}, idx::Integer, val::Float32) =
    Metal.atomic_fetch_add_explicit(ptr + (Int32(idx) - Int32(1)) * Int32(sizeof(Float32)), val)

# The matrix-unit kernel `:flash_mma`'s fragment hooks.
include("LuxTriangleAttentionMetalExt/flash_mma.jl")

end # module
