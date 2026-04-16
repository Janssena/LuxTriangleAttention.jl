rng = Random.Xoshiro(12)

function _forward_simple(q, k, v, bias, mask)
    out = similar(q)
    triangle_attention_simple!(out, q, k, v, bias, mask)
    return out
end

function _forward_tullio(q, k, v, bias, mask)
    out = similar(q)
    triangle_attention_tullio!(out, q, k, v, bias, mask)
    return out
end

function _forward_amx(q, k, v, bias, mask)
    out = similar(q)
    triangle_attention_amx!(out, q, k, v, bias, mask)
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
            ("Random mask", rand(rng, Bool[0, 1], N, N, B)),
            ("All-ones mask", ones(Bool, N, N, B)),
        );

        for (name, mask) in mask_cfg
            @testset "$name" begin
                y_simple = _forward_simple(q, k, v, bias, mask)
                y_tullio = _forward_tullio(q, k, v, bias, mask)
                y_amx = _forward_amx(q, k, v, bias, mask)

                @testset "Parity" begin
                    @test y_tullio ≈ y_simple
                    @test y_amx ≈ y_simple
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred _forward_amx(q, k, v, bias, mask)
                    @test_nowarn @inferred _forward_tullio(q, k, v, bias, mask)
                end
            end
        end
    end 
end