import Random

using Test, LuxTriangleAttention, BenchmarkTools, Metal

include("setup_python.jl");

_forward_python(q, k, v, bias, mask) =
    py"attention_reference"(q, k, v, bias, mask)

function _to_gpu(x...)
    if LuxTriangleAttention.IS_APPLE_SILICON
        return x .|> MtlArray
    else
        return x
    end
end

@testset "LuxTriangleAttention.jl" begin
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
        mask = rand(rng, Bool, N, N, B)

        q_py, k_py, v_py = to_py(q), to_py(k), to_py(v)
        bias_py = to_py(permutedims(bias, (4, 1, 2, 3)); swap_batch_dim=false)
        mask_py = isnothing(mask) ? nothing : to_py(permutedims(T.(mask), (3, 1, 2)); swap_batch_dim=false)
        
        @testset "CPU benchmarks" begin
            # compile:
            _forward_python(q_py, k_py, v_py, bias_py, mask_py)
            triangle_attention(q, k, v, bias, mask)

            println("Running CPU benchmarks for $(name)-sized input and T = Float32")
            bench_py = @benchmark $_forward_python($q_py, $k_py, $v_py, $bias_py, $mask_py)
            println("----- Python implementation -----")
            display(bench_py)
            println()

            bench_cpu = @benchmark $triangle_attention($q, $k, $v, $bias, $mask)
            println("----- CPU implementation -----")
            display(bench_cpu)
            println() 
        end

        @testset "GPU benchmarks" begin
            q_gpu, k_gpu, v_gpu, bias_gpu = _to_gpu(q, k, v, bias)
            mask_gpu = isnothing(mask) ? nothing : only(_to_gpu(mask))
            # compile:
            triangle_attention(q_gpu, k_gpu, v_gpu, bias_gpu, mask_gpu);

            println("Running GPU benchmarks for $(name)-sized input and T = Float32")
            bench_gpu = @benchmark $triangle_attention($q_gpu, $k_gpu, $v_gpu, $bias_gpu, $mask_gpu)
            println("----- GPU implementation -----")
            display(bench_gpu)
            println()
        end
    end
end