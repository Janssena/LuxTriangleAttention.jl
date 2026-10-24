# `ChannelsLast` (src/layers/channels_last.jl) against the layers it wraps: same parameters, same
# maths, channels moved to the last dimension. Pure Julia — no Python reference involved.

@testset "ChannelsLast" begin
    rng = Random.Xoshiro(11)
    C, O = 6, 5
    x = randn(rng, Float32, C, 3, 7)                  # channels-first
    cl(y) = permutedims(y, (2, 3, 1))                 # → channels-last
    x_cl = cl(x)

    @testset "Dense bias=$use_bias act=$act" for use_bias in (false, true), act in (identity, Lux.relu)
        d = Lux.Dense(C => O, act; use_bias)
        ps, st = Lux.setup(rng, d)
        ref = cl(first(d(x, ps, st)))
        @test first(ChannelsLast(d)(x_cl, ps, st)) ≈ ref
        x2 = reshape(x, C, :)                          # a lazy transpose of channels-first data
        @test first(ChannelsLast(d)(transpose(x2), ps, st)) ≈ permutedims(first(d(x2, ps, st)), (2, 1))
        @test_throws DimensionMismatch ChannelsLast(d)(x, ps, st)
    end

    @testset "LayerNorm affine=$affine" for affine in (true, false)
        ln = Lux.LayerNorm((C, 1); dims=1, affine)                 # Lux: one trailing 1 per extra dim
        ps, st = Lux.setup(rng, ln)
        affine && (ps = map(p -> p .+ randn(rng, Float32, size(p)), ps))
        @test first(ChannelsLast(ln)(x_cl, ps, st)) ≈ cl(first(ln(x, ps, st)))
    end
    @test_throws ArgumentError ChannelsLast(Lux.LayerNorm((C,)))(x_cl, (;), (;))    # dims = Colon()

    @testset "GatedLinearUnit fused=$fused $(act === Lux.swish ? "SwiGLU" : "GLU")" for fused in (true, false),
            act in (Lux.sigmoid, Lux.swish)
        g = GatedLinearUnit(C => O; fused, activation=act, use_bias=true)
        ps, st = Lux.setup(rng, g)
        @test first(ChannelsLast(g)(x_cl, ps, st)) ≈ cl(first(g(x, ps, st)))
    end

    @testset "GatedLinearUnit two inputs" begin
        g = GatedLinearUnit((4, C) => O; fused=false)
        ps, st = Lux.setup(rng, g)
        y = randn(rng, Float32, 4, 3, 7)
        @test first(ChannelsLast(g)((cl(y), x_cl), ps, st)) ≈ cl(first(g((y, x), ps, st)))
        gf = GatedLinearUnit(C => O)
        @test_throws ArgumentError ChannelsLast(gf)((x_cl, x_cl), Lux.setup(rng, gf)...)
    end

    # A narrower storage type must come back out unchanged: a stray Float32 constant (LayerNorm's
    # `epsilon` did exactly this) silently promotes the output, and a compiled 16-bit model then
    # hands its next program a Float32 array it was not traced for.
    @testset "eltype preserved: $T" for T in (Float16, Float64)
        xT = T.(x_cl)
        cast(ps) = map(p -> p isa NamedTuple ? cast(p) : T.(p), ps)
        for l in (Lux.Dense(C => O), Lux.LayerNorm((C, 1); dims=1),
                  GatedLinearUnit(C => O; fused=true), GatedLinearUnit(C => O; fused=false))
            ps, st = Lux.setup(rng, l)
            @test eltype(first(ChannelsLast(l)(xT, cast(ps), st))) === T
        end
        tm = TriangleMultiplication(C, 4)
        ps, st = Lux.setup(rng, tm)
        z = randn(rng, T, C, 5, 5, 1)
        @test eltype(first(tm(z, ones(Bool, 5, 5, 1), cast(ps), st))) === T
        @test eltype(first(tm(z, nothing, cast(ps), st))) === T
    end

    @testset "wrapper" begin
        d = Lux.Dense(C => O)
        @test Lux.parameterlength(ChannelsLast(d)) == Lux.parameterlength(d)
        @test keys(first(Lux.setup(rng, ChannelsLast(d)))) == (:weight, :bias)    # unnested
        @test_throws ArgumentError ChannelsLast(Lux.NoOpLayer())(x_cl, (;), (;))
        @test sprint(show, ChannelsLast(d)) == "ChannelsLast($(sprint(show, d)))"
    end
end
