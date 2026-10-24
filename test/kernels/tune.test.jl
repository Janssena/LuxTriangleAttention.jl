# The tuning skeleton (`src/kernels/tune.jl`) and what each built-in kernel declares it tunes
# (`flash/tune.jl`, `flash_mma/tune.jl`), with no device. Measuring a candidate (build, pipeline
# limit, verification, timing) needs one: `test/ext/metal/tune.test.jl`.

using Test, Random, LuxTriangleAttention
const LTAt = LuxTriangleAttention

@testset "tuning skeleton" begin
    caps = LTAt.DeviceCaps(32, 1024, 32768, "mock-device")
    KA, MMA = LTAt.FlashScalar(), LTAt.FlashMMA()

    @testset "grids: only what each kernel tunes" begin
        for stage in LTAt.FLASH_STAGES
            pts = LTAt.grid_points(KA, stage, 32, Float32, caps)
            @test !isempty(pts)
            # The forward also tunes the softmax sub-block; the backward stages carry no running max.
            @test keys(first(pts)) == (stage === :fwd ? (:tg, :kt, :sb) : (:tg, :kt))
            @test all(p -> p.tg % caps.warp == 0 && p.tg <= caps.max_threads, pts)
            @test all(p -> LTAt.stage_shmem(stage, 32, Float32, p) <= caps.max_shmem, pts)
            stage === :fwd && @test all(p -> p.kt % p.sb == 0, pts)
        end
        # A narrower storage type legalises larger tiles (the staged tiles are in storage precision).
        small = LTAt.DeviceCaps(32, 1024, 16384, "small")
        @test maximum(p.kt for p in LTAt.grid_points(KA, :fwd, 64, Float16, small)) >
              maximum(p.kt for p in LTAt.grid_points(KA, :fwd, 64, Float32, small))
        @test_throws ArgumentError LTAt.grid_points(KA, :nonsense, 32, Float32, caps)
        for stage in LTAt.FLASH_MMA_STAGES
            pts = LTAt.grid_points(MMA, stage, 32, Float32, caps)
            @test !isempty(pts) && keys(first(pts)) == (:nsg, :r, :tile, :ops)
            # The second products' operand type is tuned only for 16-bit storage.
            @test unique(p.ops for p in pts) == [Float32]
            for T in (Float16, LTAt.BFloat16)
                @test Set(p.ops for p in LTAt.grid_points(MMA, stage, 32, T, caps)) == Set([Float32, T])
            end
            @test all(p -> LTAt.mma_threads(p) <= caps.max_threads &&
                LTAt.mma_shmem(stage, 32, Float32, p) <= caps.max_shmem, pts)
        end
        # The device's thread cap filters before anything is compiled.
        tiny = LTAt.DeviceCaps(32, 128, 32768, "tiny")
        @test all(p -> LTAt.mma_threads(p) <= 128, LTAt.grid_points(MMA, :fwd, 32, Float32, tiny))
    end

    @testset "search: full sweep of small grids, coordinate descent of large ones" begin
        fwd = LTAt.grid_points(KA, :fwd, 32, Float32, caps)
        @test length(fwd) > LTAt.FULL_SWEEP_MAX
        # Minimum at (tg=128, kt=64, sb=4); the axes contribute independently.
        cost(p) = abs(log2(p.tg / 128)) + abs(log2(p.kt / 64)) + abs(log2(get(p, :sb, 4) / 4)) + 1.0
        seen = LTAt.search(fwd, cost)
        @test argmin(seen) == (; tg=128, kt=64, sb=4)
        @test length(seen) < length(fwd) ÷ 2               # it did not sweep everything
        # A grid within `FULL_SWEEP_MAX` is swept exhaustively.
        dq = LTAt.grid_points(KA, :dq, 32, Float32, caps)
        @test length(dq) <= LTAt.FULL_SWEEP_MAX
        @test length(LTAt.search(dq, cost)) == length(dq)
        # Undispatchable (Inf) regions are stepped over, not fatal; and if descent reaches nothing
        # finite, the whole grid is swept.
        @test isfinite(minimum(values(LTAt.search(fwd, p -> p.tg > 256 ? Inf : cost(p)))))
        lone = (; tg=32, kt=16, sb=1)
        @test argmin(LTAt.search(fwd, p -> p == lone ? 1.0 : Inf)) == lone
        @test isempty(LTAt.search(NamedTuple[], cost))
    end

    @testset "choose_config: the fastest measured point" begin
        t = Dict{NamedTuple,Float64}((; tg=256, kt=32) => 1.0, (; tg=64, kt=32) => 1.1, (; tg=32, kt=32) => Inf)
        @test LTAt.choose_config(t) == (; tg=256, kt=32)
        @test_throws ErrorException LTAt.choose_config(Dict{NamedTuple,Float64}((; tg=32) => Inf))
    end

    @testset "tune lengths: defaults and the preference" begin
        @test LTAt.tune_lengths() isa Vector{Int} && !isempty(LTAt.tune_lengths())
        @test_throws ArgumentError LTAt.set_tune_lengths!(Int[])
        @test_throws ArgumentError LTAt.set_tune_lengths!([0, 256])
    end

    @testset "calibration persistence, beside each kernel" begin
        withenv("KERNELDISPATCH_CACHE" => mktempdir()) do
            # Float16 for `:flash_mma`: its points carry a type (`ops`), stored by name.
            for (k, T) in ((KA, Float32), (MMA, Float16))
                tref = LTAt.Calibration(s => Dict{NamedTuple,Float64}(
                    first(LTAt.grid_points(k, s, 32, T, caps)) => 1e-3,
                    last(LTAt.grid_points(k, s, 32, T, caps)) => 2e-3,
                    LTAt.grid_points(k, s, 32, T, caps)[2] => Inf)
                    for s in LTAt.tune_stages(k))
                path = LTAt.calibration_path(k, caps.name, 32, T, 256)
                @test endswith(path, "__d32__$(lowercase(string(nameof(T))))__n256.toml")   # one per length
                @test startswith(path, joinpath(LTAt.KernelDispatch.cache_dir(LTAt.SDPA()),
                    String(LTAt.KernelDispatch.kernelname(k)), "calibration"))
                LTAt.save_calibration(k, path, tref)
                back = LTAt.load_calibration(k, path)
                # Finite entries only, read back identically (axis order and all).
                @test back == LTAt.Calibration(s => filter(p -> isfinite(p.second), t) for (s, t) in tref)
                # Another format version, another kernel source, or a missing stage: re-measured.
                good = read(path, String)
                write(path, replace(good, r"version = \d+" => "version = 1"))
                @test LTAt.load_calibration(k, path) === nothing
                write(path, replace(good, r"source_tag = \"[^\"]*\"" => "source_tag = \"old\""))
                @test LTAt.load_calibration(k, path) === nothing
                write(path, replace(good, "[fwd]" => "[gone]"))
                @test LTAt.load_calibration(k, path) === nothing
            end
            # The registry is KernelDispatch': `<root>/ops/LuxTriangleAttention.sdpa`.
            @test endswith(LTAt.KernelDispatch.cache_dir(LTAt.SDPA()), joinpath("ops", "LuxTriangleAttention.sdpa"))
            @test LTAt.clear_calibration!(; disk=true) === nothing
            @test !isdir(dirname(LTAt.calibration_path(KA, caps.name, 32, Float32, 256)))
        end
    end
end
