rng = Random.Xoshiro(42)

function _forward_jl(q, k, v, bias, mask)
    out = similar(q)
    triangle_attention_simple!(out, q, k, v, bias, mask)
    return out
end 

for T in [Float16, Float32, Float64]
    @testset "Precision: $T" begin
        D, H, N, B = 16, 4, 8, 2
        q = randn(rng, T, D, H, N, N, B)
        k = randn(rng, T, D, H, N, N, B)
        v = randn(rng, T, D, H, N, N, B)
        bias = randn(rng, T, H, N, N, B)

        mask_cfg = (
            ("No mask", nothing),
            ("Random mask", rand(rng, Bool, N, N, B)),
            ("All-ones mask", ones(Bool, N, N, B)),
        );

        for (name, mask) in mask_cfg
            @testset "$name" begin
                y_jl = _forward_jl(q, k, v, bias, mask)
                
                q_py, k_py, v_py = to_py(q), to_py(k), to_py(v)
                bias_py = to_py(permutedims(bias, (4, 1, 2, 3)); swap_batch_dim=false)
                mask_py = isnothing(mask) ? nothing : to_py(permutedims(T.(mask), (3, 1, 2)); swap_batch_dim=false)
                y_py = py"attention_reference"(q_py, k_py, v_py, bias_py, mask_py)

                @testset "Python parity" begin
                    @test y_jl ≈ to_jl(y_py)
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred _forward_jl(q, k, v, bias, mask)
                end
            end
        end
    end 
end