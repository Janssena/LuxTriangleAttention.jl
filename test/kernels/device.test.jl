# The backend seams in `src/kernels/device.jl`, with NO backend extension loaded.
#
# Everything here is a `Ref{Function}` a backend extension installs in its own `__init__`, or a
# `function foo end` with zero methods until one does. Run before `using Metal` appears anywhere
# in the suite (`test/kernels/runtests.jl` is included ahead of `test/ext/`), so what is under test
# is genuinely the UNINSTALLED default — a stub that fails loudly, not one that happens to have
# nothing to report. `test/ext/metal/*.test.jl` covers the seams once Metal DOES install them.

@testset "device seams, no backend loaded" begin
    @test LuxTriangleAttention.device_kernel_available() == false

    @test_throws ErrorException LuxTriangleAttention.device_binary(Float32, 32, 128, 64)
    @test_throws ErrorException LuxTriangleAttention.device_binary_bwd(Float32, 32, 128, 64)

    # Declared as `function calibrate! end` / `function _sdpa_kernel_bwd! end` — zero methods until
    # a backend extension adds one, so calling either is an ordinary dispatch failure, not a
    # message this package wrote.
    @test_throws MethodError LuxTriangleAttention.calibrate!(Float32; dim=32)
    @test_throws MethodError LuxTriangleAttention._sdpa_kernel_bwd!(
        rand(Float32, 4, 4), rand(Float32, 4, 4), rand(Float32, 4, 4),
        nothing, rand(Float32, 4, 4), rand(Float32, 4, 4), rand(Float32, 4, 4),
        rand(Float32, 4, 4))

    # `tune_flash_kernel!` checks `device_kernel_available()` itself before doing anything else, so
    # its failure here is this package's own message, not a bare `MethodError`.
    @test_throws ErrorException LuxTriangleAttention.tune_flash_kernel!(64; dim=32)
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
