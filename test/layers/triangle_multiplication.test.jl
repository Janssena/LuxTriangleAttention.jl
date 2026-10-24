# 16-bit data through the INPUT LayerNorm of `src/layers/triangle_multiplication.jl` (the contraction and
# the output LayerNorm are covered in `precision.jl`), eager on CPU, against Float64. Same σ = 60 setup as
# `triangle_attention.test.jl`. Compiled gives the same numbers and took ~24 min to compile at this
# size, so it is not repeated. Written before the fixes.

@testset "TriangleMultiplication, 16-bit, activations beyond the input LayerNorm's range" begin
    rng = Random.Xoshiro(4)
    C, H, N, B, σ = 32, 8, 8, 1, 60.0f0
    @test C * σ^2 > floatmax(Float16)
    for outgoing in (true, false)
        tm = TriangleMultiplication(C, H; is_outgoing=outgoing)
        ps, st = Lux.setup(rng, tm)
        ps16, st16 = Lux.f16(ps), Lux.f16(st)
        ps64, st64 = Lux.f64(ps16), Lux.f64(st16)
        x16 = Float16.(σ .* randn(rng, Float32, C, N, N, B) .+ 5.0f0)
        y64, _ = tm(Float64.(x16), ps64, st64)
        y16, _ = tm(x16, ps16, st16)
        @test all(isfinite, y16)
        @test relerr(y16, y64) < 5.0e-2
    end
end

# ───────────────────────────────────────────────────────────────────────────────────────────────────
# Token-major planes layout (this was `triangle_multiplication.layout.jl`)

# Pure-Julia properties of TriMulCore's planes layout (src/layers/tri_mul_core.jl) that the PyTorch
# parity tests do not cover: batch sizes 1 and 2, a `[N, N]` mask shared by every batch element, and
# batch independence (each batch element's output depends only on its own input).

@testset "TriangleMultiplication layout" begin
    rng = Random.Xoshiro(7)
    C, H, N = 16, 12, 9
    for (fused, fused_glu, use_bias) in ((true, true, false), (true, false, false), (false, false, true)),
        is_outgoing in (true, false)

        @testset "fused=$fused fused_glu=$fused_glu bias=$use_bias $(is_outgoing ? "out" : "in")" begin
            layer = TriangleMultiplication(C, H; is_outgoing, fused, fused_glu, use_bias)
            ps, st = Lux.setup(rng, layer)
            x = randn(rng, Float32, C, N, N, 2)
            mask2 = rand(rng, Bool, N, N)

            y, _ = layer(x, mask2, ps, st)
            @test size(y) == (C, N, N, 2)
            @test y ≈ first(layer(x, repeat(mask2, 1, 1, 2), ps, st))            # shared == repeated mask
            @test y[:, :, :, 2:2] ≈ first(layer(x[:, :, :, 2:2], mask2, ps, st)) # batch independence, B = 1
            @test_nowarn @inferred layer(x, mask2, ps, st)
        end
    end
end
