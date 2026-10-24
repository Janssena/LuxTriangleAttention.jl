# 16-bit data (src/precision.jl): `TriangleMultiplication` on Float16 inputs whose contraction
# overflows Float16, against a Float64 reference on the SAME 16-bit-rounded inputs and parameters.
# Pure Julia, CPU — whose Float16 matmul accumulates in Float16, so the overflow is real here too.
# That also costs ~√N·2⁻¹¹ of accuracy per contraction (a device GEMM accumulates in Float32), so the
# contraction is kept SHORT and the values large: the overflow without the accumulation error.

using LuxTriangleAttention: contraction_mode, fold_k, fold_trimul,
    FullPrecision, HalfSafe, Folded, ishalf

_rel64(y, ref) = sqrt(sum(abs2, Float64.(y) .- ref) / sum(abs2, ref))
_tof(::Type{T}, x) where {T} = x isa AbstractArray ? T.(x) : map(v -> _tof(T, v), x)

@testset "16-bit data" begin
    @testset "contraction_mode / ishalf" begin
        @test ishalf(Float16) && !ishalf(Float32) && !ishalf(Float64)
        @test contraction_mode(zeros(Float32, 2), (;)) isa FullPrecision
        @test contraction_mode(zeros(Float16, 2), (;)) isa HalfSafe
        @test contraction_mode(zeros(Float16, 2), (; fold_k=5)) == Folded(5)
        @test contraction_mode(zeros(Float32, 2), (; fold_k=5)) == Folded(5)   # the weights decide
        @test fold_k(1) == 0 && fold_k(4) == 1 && fold_k(5) == 2 && fold_k(494) == 5 && fold_k(4096) == 6
    end

    @testset "TriangleMultiplication outgoing=$out" for out in (true, false)
        rng = Random.Xoshiro(7)
        C, H, N, B = 8, 8, 32, 1
        tm = TriangleMultiplication(C, H; is_outgoing=out, use_bias=true)
        ps, st = Lux.setup(rng, tm)
        # A large positive VALUE bias makes a, b coherent (≈ 60 each), so Σₖ a·b over N = 32 reaches
        # ~1e5: past Float16's range unless the layer handles it.
        n = size(ps.core.glu_ab.linear.weight, 1) ÷ 2
        b = copy(ps.core.glu_ab.linear.bias); b[1:n] .= 120.0f0
        ps = merge(ps, (; core=merge(ps.core, (; glu_ab=merge(ps.core.glu_ab,
            (; linear=merge(ps.core.glu_ab.linear, (; bias=b))))))))
        x = randn(rng, Float32, C, N, N, B)

        # reference: Float64 on the Float16-ROUNDED inputs and parameters
        x16, ps16 = Float16.(x), _tof(Float16, ps)
        ref = Float64.(first(tm(Float64.(x16), _tof(Float64, ps16), st)))

        # the test is only meaningful if the plain Float16 contraction overflows
        a64 = 120.0 / 2                             # ≈ value · sigmoid(gate) at the bias
        @test a64^2 * N > floatmax(Float16)

        y_safe = first(tm(x16, ps16, st))           # unscaled parameters: Float32 contraction
        @test eltype(y_safe) === Float16
        @test all(isfinite, y_safe)
        @test _rel64(y_safe, ref) < 5e-3

        k = fold_k(4096)                            # a BOUND, larger than this N needs
        psf, stf = fold_trimul(ps, st, k)
        @test stf.core.fold_k == k
        y_fold, st_out = tm(x16, _tof(Float16, psf), stf)
        @test st_out.core.fold_k == k               # the state keeps it for the next call
        @test eltype(y_fold) === Float16
        @test all(isfinite, y_fold)
        # X is now STORED in Float16 (2⁻¹¹), and this test's coherent a, b give X a large common offset
        # that `layer_norm_out` subtracts, amplifying that rounding by |μ|/σ — hence a looser bound
        # than the safe path's Float32 X. (On real AlphaFold2 activations, on device: 6–7e-4.)
        @test _rel64(y_fold, ref) < 1e-2
        # The fold is EXACT: powers of two do not change Float16 rounding (absent underflow) and ε
        # is scaled with them, so a different k gives the identical result, bit for bit.
        psf2, stf2 = fold_trimul(ps, st, k + 1)
        @test first(tm(x16, _tof(Float16, psf2), stf2)) == y_fold

        # Folded parameters are valid in ANY precision: Float32 with the fold == Float32 without
        @test first(tm(x, psf, stf)) ≈ first(tm(x, ps, st)) rtol = 1e-5
    end
end

# What `src/precision.jl` gains from the Float16 work. `@test_broken` until each name exists; once it
# does, the broken test errors as an unexpected pass, which is the cue to turn it into a real test.
@testset "16-bit API still to come" begin
    has(s) = isdefined(LuxTriangleAttention, s)
    if has(:range_limited)
        @test LuxTriangleAttention.range_limited(Float16)
        @test !LuxTriangleAttention.range_limited(Float32)
        @test !LuxTriangleAttention.range_limited(Core.BFloat16)    # Float32's exponent range
        @test LuxTriangleAttention.precision_limited(Core.BFloat16)
    else
        @test_broken has(:range_limited)
    end
    @test_broken has(:fold_attention)       # 1/√d folded into the q projection (the `Folded` attention mode)
end

@testset "mask_value" begin
    mv = LuxTriangleAttention.mask_value
    @testset "$T" for T in (Float16, Float32, Float64)
        @test mv(T) isa T
        @test isfinite(mv(T))
        @test mv(T) == -floatmax(T) / 2
        @test mv(zeros(T, 3)) === mv(T)
    end
    @test mv(Float16) == Float16(-32752)

    # `Core.BFloat16` has no conversions in plain Julia; the BFloat16 that occurs in a program is
    # BFloat16s.jl's, which LuxTriangleAttention re-exports as `LuxTriangleAttention.BFloat16`.
    @testset "BFloat16 gets Float32's range" begin
        T = LuxTriangleAttention.BFloat16
        @test mv(T) isa T
        @test isfinite(Float32(mv(T)))
        @test Float32(mv(T)) ≈ mv(Float32) rtol = 1e-2          # rounded to bfloat16's 8 bits
        @test Float32(mv(T)) < -1f38
    end

    # What it replaces: the `-1e9` of OpenFold/Boltz2/AF3 is -Inf in Float16.
    @test Float16(-1e9) == -Inf
    @test isfinite(mv(Float16))

    @testset "selecting in $T: masked keys get no weight, a dead row is uniform" for T in (Float16, Float32)
        softmax(x) = (e = exp.(Float32.(x) .- maximum(Float32.(x))); e ./ sum(e))
        logits = T[3, -2, 7, 1]
        keep = [true, false, true, true]
        w = softmax(ifelse.(keep, logits, mv(T)))
        @test all(isfinite, w) && sum(w) ≈ 1
        @test w[2] == 0                                           # exactly: exp(mv - m) underflows to 0
        @test w[[1, 3, 4]] ≈ softmax(logits[[1, 3, 4]])           # the live keys are untouched
        u = softmax(ifelse.(falses(4), logits, mv(T)))            # nothing live
        @test all(isfinite, u) && all(≈(0.25f0), u)
        # the same row under -Inf is NaN, which is what a finite value avoids
        @test any(isnan, softmax(fill(T(-Inf), 4)))
    end

    @testset "as an additive bias it still fits Float16 for sensible logits" begin
        logits = Float16[-2.0e4, -1.0, 3.0e3, 2.0e4]
        @test all(isfinite, logits .+ mv(Float16))
        @test all(<(-1.0e4), logits[2:end] .+ mv(Float16))
    end
end
