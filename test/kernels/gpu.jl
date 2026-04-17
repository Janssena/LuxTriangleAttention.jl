rng = Random.Xoshiro(12)

function _forward_simple(q, k, v, bias, mask)
    out = similar(q)
    triangle_attention_simple!(out, q, k, v, bias, mask)
    return out
end

function _to_gpu(x...)
    if LuxTriangleAttention.IS_APPLE_SILICON
        return x .|> MtlArray
    else
        return x
    end
end

for T in [Float16, Float32]
    @testset "Precision: $T" begin
        D, H, N, B = 32, 6, 16, 2
        q = randn(rng, T, D, H, N, N, B)
        k = randn(rng, T, D, H, N, N, B)
        v = randn(rng, T, D, H, N, N, B)
        bias = randn(rng, T, H, N, N, B)
        q_gpu, k_gpu, v_gpu, bias_gpu = _to_gpu(q, k, v, bias)

        mask_cfg = (
            ("No mask", nothing),
            ("Random mask", rand(rng, Bool, N, N, B)),
            ("All-ones mask", ones(Bool, N, N, B)),
        );

        for (name, mask) in mask_cfg
            mask_gpu = isnothing(mask) ? nothing : only(_to_gpu(mask))

            @testset "$name" begin
                y_simple = _forward_simple(q, k, v, bias, mask)
                y_gpu = triangle_attention(q_gpu, k_gpu, v_gpu, bias_gpu, mask_gpu)

                @testset "Parity" begin
                    @test Array(y_gpu) ≈ y_simple
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred triangle_attention(q_gpu, k_gpu, v_gpu, bias_gpu, mask_gpu)
                end
            end
        end
    end 
end