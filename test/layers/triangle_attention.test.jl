# 16-bit data through `src/layers/triangle_attention.jl`, eager on CPU, against Float64 on the SAME
# Float16-rounded inputs and parameters. σ = 60 over 32 channels: Σx² ≈ 1.2e5 > floatmax(Float16), so a
# LayerNorm that accumulates its variance in the input type overflows before attention is reached.
# (Compiled by Reactant too, plain and with the rolled `chunk_size` path: `test/ext/reactant/triangle_attention.test.jl`.)

@testset "TriangleAttention, 16-bit, activations beyond the input LayerNorm's range" begin
    rng = Random.Xoshiro(3)
    C, N, B, σ = 32, 8, 1, 60.0f0
    @test C * σ^2 > floatmax(Float16)
    for starting in (true, false)
        ta = TriangleAttention(C, 8, 2; is_starting=starting)
        ps, st = Lux.setup(rng, ta)
        ps16, st16 = Lux.f16(ps), Lux.f16(st)
        ps64, st64 = Lux.f64(ps16), Lux.f64(st16)
        x16 = Float16.(σ .* randn(rng, Float32, C, N, N, B) .+ 5.0f0)
        y64, _ = ta(Float64.(x16), ps64, st64)
        y16, _ = ta(x16, ps16, st16)
        @test all(isfinite, y16)
        @test relerr(y16, y64) < 5.0e-2
    end
end
