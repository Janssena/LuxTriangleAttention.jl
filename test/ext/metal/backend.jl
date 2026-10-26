# The Metal adapter for the generic GPU tests (the files named `<file>.gpu.test.jl`), included by
# `support/backend_gpu.jl` once Metal is found and functional. The generic tests use only the names defined by the
# adapter of whichever backend was found, never a backend's own:
#
#     DeviceArray     the device array type (`MtlArray`)
#     ka_backend()    the KernelAbstractions backend object a launch context is built from
#     backend_tag()   KernelDispatch's tag for the backend
#     traced_client() the Reactant client that traces to this GPU, or `nothing` when the machine has none
#
# What is specific to Metal (the fragment lane layout, the Apple compiler's code generation) is tested in this
# folder, next to the adapter. A new backend adds a folder here with a `backend.jl`.

import KernelDispatch

const DeviceArray = Metal.MtlArray
ka_backend() = Metal.MetalBackend()
backend_tag() = KernelDispatch.MetalBackendTag()

# Reactant's default client is the CPU's, and `backend_of` answers `CPUBackend()` for a traced array under it, so the
# device-kernel route is only reached under the jax-mps client (test/mps_client.jl; `nothing` when it is absent).
function traced_client()
    isdefined(Main, :mps_client) || include(joinpath(@__DIR__, "..", "..", "mps_client.jl"))
    return Base.invokelatest(Main.mps_client)
end

# What a traced launch's custom call is named in the emitted MLIR (the traced tests assert it is there: a quiet fallback to
# the host path would pass every numerical comparison while testing nothing).
const TRACED_CALL_TARGET = "mps.metal_kernel_lib"
