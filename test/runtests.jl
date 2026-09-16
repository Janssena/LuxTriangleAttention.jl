import LuxTriangleAttention: Lux, Static.static
import Random

using Test, LuxTriangleAttention, BenchmarkTools

include("python/setup.jl");   # convert_types — shared, by inclusion order, with test/layers/

@testset "LuxTriangleAttention.jl" begin
    @testset "Layers" begin
        include("layers/runtests.jl")
    end
end