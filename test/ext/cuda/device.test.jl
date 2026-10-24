# ext/LuxTriangleAttentionCUDAExt.jl: the three device hooks the built-in kernels need from a backend.

using KernelAbstractions
import KernelDispatch
import LuxTriangleAttention as LTA

@kernel unsafe_indices = true function add_to_first!(out)
    LTA._atomic_fadd!(out, Int32(1), 1.0f0)
    LTA._atomic_fadd!(out, Int32(3), 0.5f0)
end

@testset "device hooks" begin
    be = backend_tag()

    @testset "to_device" begin
        x = rand(Float32, 3, 4)
        d = LTA.to_device(be, x)
        @test d isa DeviceArray
        @test Array(d) == x
    end

    @testset "device_caps" begin
        caps = LTA.device_caps(be)
        @test caps isa LTA.DeviceCaps
        @test caps.warp == 32
        @test caps.max_threads >= 32
        @test 0 < caps.max_shmem <= 48 * 1024            # KernelAbstractions' @localmem is static shared memory
        @test occursin(r"sm_\d+", caps.name)             # the calibration key says which architecture it is
    end

    # The `dbias` atomic, as a kernel uses it: every thread of every group adds to one element of a raw pointer.
    @testset "_atomic_fadd!" begin
        bin = KernelDispatch.ka_compile(KernelDispatch.device_backend(), add_to_first!,
            (AbstractArray{Float32},); tg=64, name="lta_test_atomic_fadd", dir=mktempdir())
        out = DeviceArray(zeros(Float32, 4))
        KernelDispatch.run_binary(bin, out; extent=(256, 1, 1))
        @test Array(out) == Float32[256, 0, 128, 0]
    end
end
