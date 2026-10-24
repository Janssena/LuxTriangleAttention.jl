# `TriMulCore` UNDER `Reactant.@compile`, in the two 16-bit contraction modes (src/precision.jl) on a
# contraction that overflows Float16, with a fully masked token. Tiny sizes: the compiled Float16 programs
# are slow to build, and the eager semantics (every projection layout, masks, batching) are pinned in
# `test/layers/tri_mul_core.test.jl`.
#
# What only a trace can get wrong:
#  - the mode is picked from `st.core.fold_k`, an `Int` that must stay a compile-time constant under
#    `to_rarray` (a traced one could not choose a method);
#  - `HalfSafe` widens the operands and `Folded` stores X in Float16, so the dispatch on a traced 16-bit
#    array (`AnyTracedRArray`) has to reach the Float32-statistics normalisation;
#  - a token whose X is all zero (a fully masked pair) must come out finite from the Float16 `Folded` path.
#
# The reference is Float64, eager, on the same 16-bit-rounded inputs and parameters.

using Reactant
using LuxTriangleAttention: TriMulCore, fold_k

@testset "TriMulCore (Reactant)" begin
    dev = Reactant.to_rarray
    C, H, N, B = 4, 3, 4, 2
    rng = Random.Xoshiro(13)
    g(l, x, mask, ps, st) = first(l(x, mask, ps, st))

    # (fused, fused_glu, use_bias) = (false, true, true) is `TriangleMultiplication(C, H; fused=false)`: the
    # separate `a`/`b` GLUs are fused GLUs, the layout `fold_trimul` once scaled the gates of.
    for out in (true, false)
        @testset "$(tmc_name((false, true, true), out))" begin
            core = TriMulCore(C, H; is_outgoing=out, fused=false, use_bias=true)
            ps0, st = Lux.setup(rng, core)
            ps = tmc_bump_values(tmc_perturb(rng, ps0), range(200.0f0, 600.0f0; length=H))
            ps16 = tmc_cast(Float16, ps)
            x16 = Float16.(randn(rng, Float32, C, N, N, B))
            # row AND column 2 of the first batch element are masked: its token (2, j) and (i, 2) have X == 0
            mask = trues(N, N, B)
            mask[2, :, 1] .= false
            mask[:, 2, 1] .= false

            ref = tmc_reference(Float64.(x16), mask, tmc_cast(Float64, ps16); outgoing=out, eps=Float64(1.0f-5))
            @test ref.Xmax > Float64(floatmax(Float16))          # the plain Float16 contraction would overflow

            @testset "HalfSafe" begin
                cf = Reactant.@compile g(core, dev(x16), dev(mask), dev(ps16), dev(st))
                y = Array(cf(core, dev(x16), dev(mask), dev(ps16), dev(st)))
                @test eltype(y) === Float16 && size(y) == size(x16)
                @test all(isfinite, y)
                @test relerr(y, ref.y) < 5e-3
            end

            @testset "Folded" begin
                k = fold_k(4096)
                psf, stf = tmc_fold(ps, st, k)
                psf16 = tmc_cast(Float16, psf)
                @test dev(stf).fold_k === k                       # an Int stays an Int: the mode is static
                cf = Reactant.@compile g(core, dev(x16), dev(mask), dev(psf16), dev(stf))
                y = Array(cf(core, dev(x16), dev(mask), dev(psf16), dev(stf)))
                @test eltype(y) === Float16 && size(y) == size(x16)
                @test all(isfinite, y)                           # including the fully masked tokens
                @test relerr(y, ref.y) < 5e-3
            end
        end
    end
end
