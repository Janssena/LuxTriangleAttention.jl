import LuxTriangleAttention: Lux, Static.static
import Random

using Test, LuxTriangleAttention, BenchmarkTools

include("python/setup.jl");   # convert_types — shared, by inclusion order, with test/layers/

@testset "LuxTriangleAttention.jl" begin
    @testset "Kernels" begin
        include("kernels/runtests.jl")
    end

    @testset "Layers" begin
        include("layers/runtests.jl")
    end

    @testset "Extensions" begin
        include("ext/runtests.jl")
    end
end