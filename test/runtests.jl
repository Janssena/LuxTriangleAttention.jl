import LuxTriangleAttention: Lux, Static.static
import Random

using Test, LuxTriangleAttention, BenchmarkTools

include("python/setup.jl");

@testset "LuxTriangleAttention.jl" begin
    @testset "Layers" begin
        include("layers/runtests.jl")
    end
end