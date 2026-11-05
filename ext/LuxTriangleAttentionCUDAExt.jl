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
using KernelDispatch: CUDABackendTag

const LTA = LuxTriangleAttention

# The one thing `flash_bwd_dq!` needs that KernelAbstractions cannot express: an atomic add on the raw pointer
# `dbias` (see `_atomic_fadd!`'s docstring in `flash/backward.jl`). `idx` is 1-based, in elements; the byte offset
# is Int64, so no conversion here can throw (a kernel whose code can throw is refused at build time on CUDA).
LTA._atomic_fadd!(ptr::Core.LLVMPtr{Float32,CUDACore.AS.Global}, idx::Integer, val::Float32) =
    CUDACore.atomic_add!(ptr + (Int(idx) - 1) * sizeof(Float32), val)

# `TriMul`'s tile and block sizes, tuned on an NVIDIA A40 (N = 384 and 768, C = H = 128, test/bench/kernels/trimul_tune.bench.jl).
# Only `nb` matters. Against the generic 16384 (N = 768 / 1024, forward, backward): 131072 is 29% / 27% faster forward
# and 20% / 19% faster backward; 262144 and 524288 add only 3% and 5% more forward, 4% and 6% backward, for 2x and 4x the
# workspace; 4096 is 2x slower. `nb` is capped at the token count N^2, so this changes nothing below N = 512. `TB`, `NT`
# and `NTE` move the time by less than the run-to-run noise (<= 3%), so they keep the generic defaults. The workspace is
# about 4 KB per token of `nb` at C = H = 128 (the `[nb, 4H]` projection dominates), i.e. ~1 GB at 262144.
LTA.trimul_tunables(::CUDABackendTag) = (TB=32, NT=256, NTE=256, nb=262144)

# The matrix-unit kernel `:flash_mma`'s fragment hooks.
include("LuxTriangleAttentionCUDAExt/flash_mma.jl")

end # module
