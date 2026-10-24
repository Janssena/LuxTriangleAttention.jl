# `resolve_defaults` (src/utils.jl): how a layer's `use_bias`-style option becomes one value per key.

using LuxTriangleAttention: resolve_defaults

@testset "resolve_defaults" begin
    keys3 = (:a, :b, :c)

    @testset "a Bool applies to every key" begin
        @test resolve_defaults(true, keys3) == (a=true, b=true, c=true)
        @test resolve_defaults(false, keys3) == (a=false, b=false, c=false)
    end

    @testset "a NamedTuple is taken as it is" begin
        nt = (a=true, b=false, c=true)
        @test resolve_defaults(nt, keys3) === nt
    end

    @testset "(fallback, overrides): the overrides win, the rest take the fallback" begin
        @test resolve_defaults((false, (b=true,)), keys3) == (b=true, a=false, c=false)
        @test resolve_defaults((true, (a=false, c=false)), keys3) == (a=false, c=false, b=true)
        @test resolve_defaults((true, (;)), keys3) == (a=true, b=true, c=true)
        @test resolve_defaults((true, (a=false, b=false, c=false)), keys3) == (a=false, b=false, c=false)
    end

    @testset "type stability, with the keys a literal as in the layer constructors" begin
        # (The keys are values, not types, so inference needs them as constants of the calling function.)
        bool() = resolve_defaults(true, (:a, :b, :c))
        overrides() = resolve_defaults((false, (b=true,)), (:a, :b, :c))
        @test_nowarn @inferred bool()
        @test_nowarn @inferred overrides()
    end
end
