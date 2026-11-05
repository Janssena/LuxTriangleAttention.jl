module LuxTriangleAttention

import KernelDispatch
import Lux, Random
import LinearAlgebra
import Preferences

using KernelAbstractions
using KernelDispatch: OutArray, call_binary, backend_of
using KernelDispatch: KernelBackend, CPUBackend, MetalBackendTag, CUDABackendTag, ROCmBackendTag,
    UnknownBackend, GPUBackend, device_dense, traced_autodiff, KERNEL_IN_TRACED_AUTODIFF, realtype
using KernelDispatch: DeviceCaps, device_caps, to_device, release!
using Static
using TOML
using BFloat16s: BFloat16

include("utils.jl")
export resolve_defaults

include("layers/primitives.jl")
export LayerNormNoBias, StableLayerNorm

include("layers/glu.jl")
export GatedLinearUnit, SwiGLU

include("layers/channels_last.jl")
export ChannelsLast

include("precision.jl")
public ishalf, mask_value, fold_k, fold_trimul, contraction_mode, layernorm_f32stats
public ContractionMode, FullPrecision, HalfSafe, Folded

include("kernels/sdpa_op.jl")
include("kernels/stage_args.jl")
export SDPA, list_kernels, use_kernel!, current_kernel, RESERVED_KERNEL_NAME
export backend_of
public KERNEL_IN_TRACED_AUTODIFF

include("kernels/flash/forward.jl")
include("kernels/flash/backward.jl")
include("kernels/flash_mma/forward.jl")
include("kernels/flash_mma/backward.jl")
include("kernels/trimul/kernels.jl")

include("kernels/device.jl")

include("kernels/verify.jl")
include("kernels/launch.jl")
include("kernels/tune.jl")
include("kernels/flash/tune.jl")
include("kernels/flash_mma/tune.jl")
export calibrate!, clear_calibration!, tune_flash_kernel!, tune_flash_mma!, calibrate_mma!
export tune_lengths, set_tune_lengths!
public OutArray

include("layers/attention.jl")
include("layers/triangle_attention.jl")
export Attention, TriangleAttention, prep_mask, prep_bias

include("layers/chunk.jl")
export chunked_compile_options, check_rolled

include("layers/triangle_multiplication.jl")
include("layers/tri_mul_core.jl")
export TriangleMultiplication, TriMulCore

include("kernels/trimul/op.jl")
include("kernels/trimul/backward.jl")
export TriMul
public TriMulStaged, trimul_tunables, set_trimul_tunables!, trimul_contraction_type

"""
    __init__()

Register the kernels this package builds itself with KernelDispatch: the two attention kernels (`:default` and
`:flash_mma`) and triangle multiplication's `:default`. KernelDispatch loads their variants from the cache —
dropping any built from other kernel sources. A missing variant is tuned (attention) or compiled (triangle
multiplication) on demand. Which one runs is each op's `@default_kernel` (the first, or the package preference
`kernel.sdpa` / `kernel.trimul`), switched for a session with `KernelDispatch.use_kernel!`.
"""
function __init__()
    KernelDispatch.register_kernel!(SDPA(), FlashScalar())
    KernelDispatch.register_kernel!(SDPA(), FlashMMA())
    KernelDispatch.register_kernel!(TriMul(), TriMulStaged())
    return nothing
end

end
