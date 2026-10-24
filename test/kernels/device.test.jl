# The backend hooks in `src/kernels/device.jl`, with NO backend extension loaded.
#
# Each hook is a method on KernelDispatch' backend tags that a backend extension adds for its own tag.
# Run before `using Metal` appears anywhere in the suite (it is the first unit of `test/kernels/runtests.jl`, and a
# `cpu` or `device` run never loads a backend at all), so what is under test is genuinely the UNINSTALLED default — an error that
# names what to load, not a stub that happens to have nothing to report. `test/ext/metal/`
# covers the hooks once Metal adds them.

@testset "device hooks, no backend loaded" begin
    L = LuxTriangleAttention
    @test L.device_kernel_available() == false
    tag = L.KernelDispatch.MetalBackendTag()
    @test_throws ArgumentError L.device_caps(tag)
    @test_throws ArgumentError L.to_device(tag, zeros(Float32, 2))

    # Building needs the one loaded device backend: none, so KernelDispatch refuses.
    for stage in L.FLASH_STAGES
        @test_throws ArgumentError L.build_stage(L.FlashScalar(), stage, Float32, 32, (; tg=128, kt=64, sb=1))
    end

    # Calibrating and tuning check `device_kernel_available()` first: this package's own message.
    @test_throws ErrorException L.calibrate!(Float32; dim=32)
    @test_throws ErrorException L.calibrate_mma!(Float32; dim=32)
    @test_throws ErrorException L.tune_flash_kernel!(64; dim=32)
    @test_throws ErrorException L.tune_flash_mma!(64; dim=32)
end

@testset "kernel_source_tag" begin
    tag = LuxTriangleAttention.kernel_source_tag()
    @test tag isa String
    @test occursin(r"^[0-9a-z]+$", tag)         # base-36, see its docstring
    @test LuxTriangleAttention.kernel_source_tag() == tag   # stable within one process

    # Computed once at precompile time over `flash/forward.jl` and `flash/backward.jl`; changing
    # either changes the tag, which is the entire reason it exists — see its docstring for the
    # stale-binary bug this catches. Not re-derived here (would just re-hash the same files), only
    # pinned as non-trivial: the source files are non-empty, so this is not the hash of nothing.
    @test !isempty(tag)
end

@testset "clear_calibration! needs no backend" begin
    # Purely in-memory / filesystem bookkeeping — no device required.
    @test LuxTriangleAttention.clear_calibration!() === nothing
    @test LuxTriangleAttention.clear_calibration!(; disk=false) === nothing
end
