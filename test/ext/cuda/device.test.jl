# ext/LuxTriangleAttentionCUDAExt.jl: the device hook the built-in kernels need from CUDA. (`to_device`,
# `device_caps` and `release!` are KernelDispatch's now, and tested there.)

using KernelAbstractions
import KernelDispatch
import LuxTriangleAttention as LTA

@kernel unsafe_indices = true function add_to_first!(out)
    LTA._atomic_fadd!(out, Int32(1), 1.0f0)
    LTA._atomic_fadd!(out, Int32(3), 0.5f0)
end

@testset "device hooks" begin
    # The `dbias` atomic, as a kernel uses it: every thread of every group adds to one element of a raw pointer.
    @testset "_atomic_fadd!" begin
        bin = KernelDispatch.ka_compile(KernelDispatch.device_backend(), add_to_first!,
            (AbstractArray{Float32},); tg=64, name="lta_test_atomic_fadd", dir=mktempdir())
        out = DeviceArray(zeros(Float32, 4))
        KernelDispatch.run_binary(bin, out; extent=(256, 1, 1))
        @test Array(out) == Float32[256, 0, 128, 0]
    end
end
