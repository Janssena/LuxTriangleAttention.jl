import Random

using Test, TriangleAttention, BenchmarkTools

include("setup_python.jl");

@testset "TriangleAttention.jl" begin
    @testset "Python parity check" begin
        # This checks that our triangle_attention_simple! is correct so that we 
        # can use it instead of the python function for downstream parity checks
        include("kernels/simple.jl")
    end

    @testset "cpu" begin
        include("kernels/cpu.jl")
    end
end