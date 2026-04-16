import Random

using Test, TriangleAttention, BenchmarkTools

include("setup_python.jl");

_forward_python(q, k, v, bias, mask) =
    py"attention_reference"(q, k, v, bias, mask)

@testset "TriangleAttention.jl" begin
    # setup
    rng = Random.Xoshiro(42)
    T = Float32
    dim_cfg = (
        ("small", (D = 16, H = 4, N = 32, B = 2)),
        ("medium", (D = 16, H = 8, N = 256, B = 2))
    )
    
    for (name, dims) in dim_cfg
        D, H, N, B = values(dims)
        q = randn(rng, T, D, H, N, N, B)
        k = randn(rng, T, D, H, N, N, B)
        v = randn(rng, T, D, H, N, N, B)
        bias = randn(rng, T, H, N, N, B)
        mask = rand(rng, T[0, 1], N, N, B)

        q_py, k_py, v_py = to_py(q), to_py(k), to_py(v)
        bias_py = to_py(permutedims(bias, (4, 1, 2, 3)); swap_batch_dim=false)
        mask_py = isnothing(mask) ? nothing : to_py(permutedims(mask, (3, 1, 2)); swap_batch_dim=false)
        
        @testset "CPU benchmarks" begin
            println("Running CPU benchmarks for $(name)-sized input and T = Float32")
            bench_py = @benchmark $_forward_python($q_py, $k_py, $v_py, $bias_py, $mask_py)
            println("----- Python implementation -----")
            display(bench_py)
            println()

            bench_tullio = @benchmark $triangle_attention($q, $k, $v, $bias, $mask)
            println("----- Tullio implementation -----")
            display(bench_tullio)
            println() 
        end
    end
end