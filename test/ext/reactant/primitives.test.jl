# `StableLayerNorm` and `LayerNormNoBias` (src/layers/primitives.jl) traced by Reactant: Float32 programs are the
# wrapped layer's own, Float16 gets Float32 statistics. The eager tests are `primitives.test.jl`; this is its own file
# because `@compile` cannot be parsed without Reactant loaded.

import Reactant

@testset "traced (Reactant)" begin
    LN = StableLayerNorm
    C, N, B, σ = 32, 6, 2, 60.0f0
    rng = Random.Xoshiro(11)
    x16 = Float16.(σ .* randn(rng, Float32, C, N, N, B) .+ 5.0f0)
    f64(x) = Float64.(x)
    dev = Reactant.to_rarray
    ln = LN((C, 1, 1); dims=1)
    ps, st = Lux.setup(rng, ln)
    g(l, x, ps, st) = first(l(x, ps, st))
    # Float32 is the wrapped layer's own program: the same HLO as `Lux.LayerNorm`, so any `AnyTracedRArray`
    # method Lux/LuxLib has for it is hit unchanged.
    x32 = Float32.(x16)
    hlo(l) = string(Reactant.@code_hlo g(l, dev(x32), dev(ps), dev(st)))
    @test hlo(ln) == hlo(ln.layer)
    # Float16 traced: Float32 statistics, finite and accurate
    ps16, st16 = Lux.f16(ps), Lux.f16(st)
    cf = Reactant.@compile g(ln, dev(x16), dev(ps16), dev(st16))
    y = Array(cf(ln, dev(x16), dev(ps16), dev(st16)))
    y64, _ = ln.layer(f64(x16), Lux.f64(ps16), Lux.f64(st16))
    @test eltype(y) === Float16 && all(isfinite, y) && isapprox(y, y64; rtol=5e-3)
    # the frozen layer traced too
    nb = LayerNormNoBias((C, 1, 1); dims=1)
    pn, sn = Lux.setup(rng, nb)
    pn16, sn16 = Lux.f16(pn), Lux.f16(sn)
    cn = Reactant.@compile g(nb, dev(x16), dev(pn16), dev(sn16))
    yn = Array(cn(nb, dev(x16), dev(pn16), dev(sn16)))
    yn64, _ = nb.layer(f64(x16), Lux.f64(pn16), Lux.f64(sn16))
    @test eltype(yn) === Float16 && all(isfinite, yn) && isapprox(yn, yn64; rtol=5e-3)
end

