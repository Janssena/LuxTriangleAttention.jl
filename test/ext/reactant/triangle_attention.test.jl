# `TriangleAttention` compiled by Reactant at 16 bits, plain and with the rolled `chunk_size` path (the production-shaped one), against
# Float64 on the SAME Float16-rounded inputs and parameters, as the eager `test/layers/triangle_attention.test.jl` does. The input
# is large enough that a LayerNorm accumulating its variance in Float16 overflows before attention is reached.

using Reactant

@testset "TriangleAttention, 16-bit, compiled, $name" for (name, kw) in (("plain", (;)), ("chunk_size=4", (; chunk_size=4)))
    rng = Random.Xoshiro(3)
    C, N, B, σ = 32, 8, 1, 60.0f0
    ta = TriangleAttention(C, 8, 2; kw...)
    ps, st = Lux.setup(rng, ta)
    ps16, st16 = Lux.f16(ps), Lux.f16(st)
    ps64, st64 = Lux.f64(ps16), Lux.f64(st16)
    x16 = Float16.(σ .* randn(rng, Float32, C, N, N, B) .+ 5.0f0)
    y64, _ = ta(Float64.(x16), ps64, st64)
    dev = Reactant.to_rarray
    h(l, x, ps, st) = first(l(x, ps, st))
    args = (ta, dev(x16), dev(ps16), dev(st16))
    y16 = Array(Reactant.compile(h, args)(args...))
    @test eltype(y16) === Float16
    @test all(isfinite, y16)
    @test relerr(y16, y64) < 5.0e-2
end
