module LuxTriangleAttention

import Lux, Random

using KernelAbstractions
using Static
using TOML

# Element type behind a possibly-wrapped number type (`TracedRNumber{Float32}` → `Float32`);
# extended by the Reactant extension.
_realtype(::Type{T}) where {T<:Number} = T

include("utils.jl")
export resolve_defaults

include("layers/primitives.jl")
export LayerNormNoBias

include("layers/glu.jl")
export GatedLinearUnit, SwiGLU

include("layers/channels_last.jl")
export ChannelsLast

# Attention kernels, opted into per layer with `Attention(...; use_kernel=static(true))`. Included
# before the layers so the kernels are in scope for the `_sdpa` dispatch.
# `registry.jl` first: `backend.jl`'s `_ensure_variant` names `KernelEntry` in its SIGNATURE, which
# has to exist at definition time. The reverse direction is safe — `registry.jl` reaches
# `kernel_cache_root` only from function bodies, which resolve at call time.
include("kernels/registry.jl")
export register_kernel!, unregister_kernel!, list_kernels, use_kernel!, current_kernel
export LTA_KERNEL, RESERVED_KERNEL_NAME
include("kernels/backend.jl")
export backend_of, kernel_cache_root
# The device kernel and its tuning policy are ordinary source, not an extension: triangle attention
# is a device operation in essentially every real use, so KernelAbstractions is a dependency rather
# than something to discover at load time. This also removes the module-registry indirection that
# previously existed only because one extension cannot import another.
include("kernels/flash/forward.jl")
include("kernels/flash/backward.jl")
include("kernels/flash/tune.jl")
export calibrate!, clear_calibration!, tune_flash_kernel!
include("kernels/device.jl")
export device_kernel_available

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

Rehydrate the kernel registry from the on-disk cache.

A kernel registered in an earlier session is available immediately, without the user re-running
`register_kernel!`. The SELECTION is not restored — see `LTA_KERNEL` for why a stale choice must not
follow the user into an unrelated session.
"""
function __init__()
    load_registry!()
    return nothing
end

end
