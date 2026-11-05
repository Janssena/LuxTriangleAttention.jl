# `TriangleMultiplication(use_kernel=true)` UNDER `Reactant.@compile`: the staged kernel is an eager pipeline (its block loops write
# workspaces in place, which jax-mps cannot trace, and the traced layer is the faster program anyway; see docs/FEATURE_trimul_kernel.md),
# so a trace must compile the layer's own forward, to the same numbers as `use_kernel=false` and with no kernel in the IR.

using Reactant
using LuxTriangleAttention: TriangleMultiplication

@testset "TriangleMultiplication(use_kernel=true) (Reactant)" begin
    dev = Reactant.to_rarray
    C, H, N, B = 8, 6, 5, 2
    rng = Random.Xoshiro(21)
    g(l, x, mask, ps, st) = first(l(x, mask, ps, st))
    for out in (true, false)
        lk = TriangleMultiplication(C, H; is_outgoing=out, use_bias=true, use_kernel=true)
        lh = TriangleMultiplication(C, H; is_outgoing=out, use_bias=true)
        ps, st = Lux.setup(rng, lh)
        x = randn(rng, Float32, C, N, N, B)
        mask = rand(rng, N, N, B) .< 0.8
        args = (dev(x), dev(mask), dev(ps), dev(st))
        ck = Reactant.@compile g(lk, args...)
        ch = Reactant.@compile g(lh, args...)
        yk, yh = Array(ck(lk, args...)), Array(ch(lh, args...))
        @test size(yk) == size(x)
        @test yk ≈ yh rtol = 1.0f-6
        @test yk ≈ first(lh(x, mask, ps, st)) rtol = 1.0f-4
        # no kernel custom call in the program
        @test !occursin("trimul", string(Reactant.@code_hlo g(lk, args...)))
    end
end
