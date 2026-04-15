rng = Random.Xoshiro(12)

function _forward_simple(q, k, v, bias, mask)
    out = similar(q)
    triangle_attention_simple!(out, q, k, v, bias, mask)
    return out
end

for T in [Float16, Float32, Float64]
    @testset "Precision: $T" begin
        D, H, N, B = 32, 6, 16, 2
        q = randn(rng, T, D, H, N, N, B)
        k = randn(rng, T, D, H, N, N, B)
        v = randn(rng, T, D, H, N, N, B)
        bias = randn(rng, T, H, N, N, B)

        mask_cfg = (
            ("No mask", nothing),
            ("Random mask", rand(rng, T[0, 1], N, N, B)),
            ("All-ones mask", ones(T, N, N, B)),
        );

        for (name, mask) in mask_cfg
            @testset "$name" begin
                y_simple = _forward_simple(q, k, v, bias, mask)
                y_tullio = triangle_attention(q, k, v, bias, mask)

                @testset "Parity" begin
                    @test y_tullio ≈ y_simple
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred triangle_attention(q, k, v, bias, mask)
                end

                if name == "Random mask" && T == Float32
                    bench_simple = @benchmark $_forward_simple($q, $k, $v, $bias, $mask)
                    println("----- Simple julia implementation ($name, $T) -----")
                    display(bench_simple)
                    println()

                    bench_tullio = @benchmark $triangle_attention($q, $k, $v, $bias, $mask)
                    println("----- Tullio implementation ($name, $T) -----")
                    display(bench_tullio)
                    println()
                end
            end
        end
    end 
end