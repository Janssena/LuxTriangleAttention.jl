# `src/kernels/flash_mma/tune.jl`: the entry point's argument checks. (Calibration and binary builds
# are exercised end to end by the forward and backward tests, which tune lazily.)

@testset "head dim not a multiple of 8 is refused" begin
    @test_throws ArgumentError tune_flash_mma!(16; dim=12)
end
