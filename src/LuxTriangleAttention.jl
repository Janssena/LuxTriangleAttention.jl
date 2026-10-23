module LuxTriangleAttention

import KernelDispatch
import Lux, Random
import Preferences

using KernelAbstractions
using KernelDispatch: OutArray, call_binary, backend_of
using KernelDispatch: KernelBackend, CPUBackend, MetalBackendTag, CUDABackendTag, ROCmBackendTag,
    UnknownBackend, GPUBackend, device_dense, traced_autodiff, KERNEL_IN_TRACED_AUTODIFF, realtype
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

include("kernels/sdpa_op.jl")
include("kernels/stage_args.jl")
export SDPA, list_kernels, use_kernel!, current_kernel, RESERVED_KERNEL_NAME
export backend_of
public KERNEL_IN_TRACED_AUTODIFF

include("kernels/flash/forward.jl")
include("kernels/flash/backward.jl")
include("kernels/flash_mma/forward.jl")
include("kernels/flash_mma/backward.jl")

include("kernels/device.jl")
export device_kernel_available

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

"""
    __init__()

Register the two attention kernels this package builds itself (`:default` and `:flash_mma`) with
KernelDispatch, which loads their tuned variants from the cache — dropping any built from other kernel
sources. A missing variant is tuned on demand. Which one runs is `SDPA`'s `@default_kernel` (the first,
or the package preference `kernel.sdpa`), switched for a session with `use_kernel!`.
"""
function __init__()
    _register_builtin_kernels!()
    return nothing
end

end
