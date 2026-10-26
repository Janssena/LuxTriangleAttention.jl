# The CUDA adapter for the generic GPU tests (the files named `<file>.gpu.test.jl`), included by
# `support/backend_gpu.jl` once CUDA is found and functional. Same contract as `test/ext/metal/backend.jl`:
#
#     DeviceArray     the device array type (`CuArray`)
#     ka_backend()    the KernelAbstractions backend object a launch context is built from
#     backend_tag()   KernelDispatch's tag for the backend
#     traced_client() the Reactant client that traces to this GPU, or `nothing` when the machine has none
#
# What is specific to CUDA (the tensor-core fragment layout, the PTX the hooks emit) is tested in this folder.

import KernelDispatch

const DeviceArray = GPU_PKG.CuArray
ka_backend() = GPU_PKG.CUDABackend()
backend_tag() = KernelDispatch.CUDABackendTag()

# Reactant's own CUDA client; KernelDispatch's tests name it the same way.
traced_client() = "gpu"

# What a traced launch's custom call is named in the emitted MLIR (XLA's kernel-call handler; see KernelDispatch's
# `ext/KernelDispatchReactantExt/cuda.jl`). The traced tests assert it is there: a quiet fallback to the host path would
# pass every numerical comparison while testing nothing.
const TRACED_CALL_TARGET = "triton_kernel_call_ffi"
