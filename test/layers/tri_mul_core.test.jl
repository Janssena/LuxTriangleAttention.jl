# `TriMulCore` (src/layers/tri_mul_core.jl): the token-major planes layout against an independent
# per-element reference, `fold_trimul` for every projection layout, and the three contraction modes
# (`FullPrecision`, `HalfSafe`, `Folded`; src/precision.jl). Pure Julia on CPU. Float16 matmul there
# accumulates in Float16, so the contractions are kept short and the values large: the overflow without
# the accumulation error.
#
# Every configuration is `(fused, fused_glu, use_bias)`: one projection for `a` and `b` or one each,
# each a fused GLU (one Dense, values then gates) or a GLU with its own gate Dense.

using LuxTriangleAttention: TriMulCore, contraction_mode, fold_trimul, fold_k, FullPrecision, HalfSafe,
    Folded, _apply_tri_mask
using LuxTriangleAttention: BFloat16 as TMCBF16

# ── tests ─────────────────────────────────────────────────────────────────────────────────────────

@testset "TriMulCore" begin
    C, H, N, B = 7, 5, 8, 2
    eps = 0.1f0        # large, so that the output LayerNorm's ε matters (and `Folded` must scale it)

    @testset "layout vs a per-element reference: $(tmc_name(cfg, out))" for cfg in TMC_CONFIGS, out in (true, false)
        fused, fused_glu, use_bias = cfg
        rng = Random.Xoshiro(hash((cfg, out)))
        core = TriMulCore(C, H; is_outgoing=out, fused, fused_glu, use_bias, layernorm_eps=eps)
        ps0, st = Lux.setup(rng, core)
        ps = Lux.f64(tmc_perturb(rng, ps0))
        x = randn(rng, C, N, N, B)
        for mask in (nothing, rand(rng, Bool, N, N), rand(rng, Bool, N, N, B))
            y, st2 = core(x, mask, ps, st)
            ref = tmc_reference(x, mask, ps; outgoing=out, eps=Float64(eps))
            @test size(y) == (C, N, N, B)
            @test y ≈ ref.y rtol = 1e-10
            @test st2 === st
        end
    end

    @testset "fold_trimul structure: $(tmc_name(cfg, true))" for cfg in TMC_CONFIGS
        fused, fused_glu, use_bias = cfg
        rng = Random.Xoshiro(hash(cfg))
        core = TriMulCore(C, H; fused, fused_glu, use_bias)
        ps0, st0 = Lux.setup(rng, core)
        ps = tmc_perturb(rng, ps0)
        ps_before = deepcopy(ps)

        # exactly the VALUE rows of every `a`/`b` projection (weight and bias) are scaled; gates are not
        function check_glu(g0, g1, s)
            n = isempty(g0.gate) ? size(g0.linear.weight, 1) ÷ 2 : size(g0.linear.weight, 1)
            @test g1.linear.weight[1:n, :] == s .* g0.linear.weight[1:n, :]
            @test g1.linear.weight[(n + 1):end, :] == g0.linear.weight[(n + 1):end, :]
            if haskey(g0.linear, :bias)
                @test g1.linear.bias[1:n] == s .* g0.linear.bias[1:n]
                @test g1.linear.bias[(n + 1):end] == g0.linear.bias[(n + 1):end]
            end
            @test g1.gate == g0.gate
        end

        for k in (1, 4)
            s = 2.0f0^(-k)
            psf, stf = tmc_fold(ps, st0, k)
            if fused
                check_glu(ps.glu_ab, psf.glu_ab, s)
                @test isempty(psf.glu_a) && isempty(psf.glu_b)
            else
                check_glu(ps.glu_a, psf.glu_a, s)
                check_glu(ps.glu_b, psf.glu_b, s)
                @test isempty(psf.glu_ab)
            end
            @test psf.layer_norm_out == ps.layer_norm_out        # everything else is untouched
            @test psf.glu_out == ps.glu_out
            @test stf == merge(st0, (; fold_k=k))
            @test ps == ps_before                                # out of place
        end

        # k = 0 changes nothing but the mode; the scaling is exact in any precision
        ps0f, _ = tmc_fold(ps, st0, 0)
        @test ps0f == ps
        @test tmc_cast(Float16, first(tmc_fold(ps, st0, 3))) == first(tmc_fold(tmc_cast(Float16, ps), st0, 3))
    end

    @testset "fold_trimul on a TriangleMultiplication leaves its input LayerNorm alone" begin
        rng = Random.Xoshiro(5)
        tm = TriangleMultiplication(C, H; fused=false)       # the default `fused_glu`: glu_a/glu_b are fused GLUs
        ps, st = Lux.setup(rng, tm)
        ps = tmc_perturb(rng, ps)
        psf, stf = fold_trimul(ps, st, 3)
        @test psf.layer_norm == ps.layer_norm
        @test stf.core.fold_k == 3 && stf.layer_norm == st.layer_norm
    end

    @testset "folded == unfolded, Float64: $(tmc_name(cfg, out))" for cfg in TMC_CONFIGS, out in (true, false)
        fused, fused_glu, use_bias = cfg
        rng = Random.Xoshiro(hash((cfg, out, 2)))
        core = TriMulCore(C, H; is_outgoing=out, fused, fused_glu, use_bias, layernorm_eps=eps)
        ps0, st = Lux.setup(rng, core)
        ps = Lux.f64(tmc_perturb(rng, ps0))
        x = randn(rng, C, N, N, B)
        for mask in (nothing, rand(rng, Bool, N, N), rand(rng, Bool, N, N, B)), k in (0, 2, 5)
            y, _ = core(x, mask, ps, st)
            psf, stf = tmc_fold(ps, st, k)
            @test contraction_mode(x, stf) == Folded(k)
            yf, st2 = core(x, mask, psf, stf)
            @test st2 === stf                                    # `fold_k` stays for the next call
            # (the folded output LayerNorm takes its statistics in Float32, hence not 1e-10)
            @test relerr(yf, y) < 1e-5
        end
    end

    @testset "16-bit data, contraction beyond Float16: $(tmc_name(cfg, out))" for cfg in TMC_CONFIGS, out in (true, false)
        fused, fused_glu, use_bias = cfg
        use_bias || continue                                     # the overflow comes from large value biases
        rng = Random.Xoshiro(hash((cfg, out, 3)))
        core = TriMulCore(C, H; is_outgoing=out, fused, fused_glu, use_bias, layernorm_eps=1.0f-5)
        ps0, st = Lux.setup(rng, core)
        ps = tmc_bump_values(tmc_perturb(rng, ps0), range(100.0f0, 400.0f0; length=H))
        x16 = Float16.(randn(rng, Float32, C, N, N, B))
        ps16 = tmc_cast(Float16, ps)
        ps64 = tmc_cast(Float64, ps16)                           # the same numbers, wide
        x64 = Float64.(x16)

        for mask in (nothing, rand(rng, Bool, N, N, B))
            ref = tmc_reference(x64, mask, ps64; outgoing=out, eps=Float64(1.0f-5))
            # the case is only meaningful if the plain Float16 contraction overflows
            @test ref.Xmax > 2 * Float64(floatmax(Float16))

            @testset "HalfSafe" begin
                @test contraction_mode(x16, st) isa HalfSafe
                y, st2 = core(x16, mask, ps16, st)
                @test eltype(y) === Float16 && size(y) == size(x16)
                @test all(isfinite, y)
                @test relerr(y, ref.y) < 5e-3
                @test st2 === st
            end

            @testset "Folded" begin
                k = fold_k(4096)                                 # a bound, larger than this N needs
                psf, stf = tmc_fold(ps, st, k)
                psf16 = tmc_cast(Float16, psf)
                @test contraction_mode(x16, stf) == Folded(k)
                y, st2 = core(x16, mask, psf16, stf)
                @test eltype(y) === Float16 && size(y) == size(x16)
                @test all(isfinite, y)
                @test relerr(y, ref.y) < 5e-3                    # X is stored in Float16 here
                @test st2 === stf
                # powers of two do not change Float16 rounding (absent underflow, ε scales with them):
                # another k gives the identical result, bit for bit
                psf2, stf2 = tmc_fold(ps, st, k + 1)
                @test first(core(x16, mask, tmc_cast(Float16, psf2), stf2)) == y
            end
        end
    end

    @testset "BFloat16 takes the same paths" begin
        rng = Random.Xoshiro(17)
        core = TriMulCore(C, H; use_bias=true)
        ps0, st = Lux.setup(rng, core)
        ps = tmc_bump_values(tmc_perturb(rng, ps0), range(100.0f0, 400.0f0; length=H))
        xb = TMCBF16.(randn(rng, Float32, C, N, N, B))
        psb = tmc_cast(TMCBF16, ps)
        ref = tmc_reference(Float64.(xb), nothing, tmc_cast(Float64, psb); outgoing=true, eps=Float64(1.0f-5))

        @test contraction_mode(xb, st) isa HalfSafe
        y, _ = core(xb, nothing, psb, st)
        @test eltype(y) === TMCBF16 && all(isfinite, Float32.(y))
        @test relerr(Float32.(y), ref.y) < 5e-2

        psf, stf = tmc_fold(ps, st, 6)
        @test contraction_mode(xb, stf) == Folded(6)
        yf, _ = core(xb, nothing, tmc_cast(TMCBF16, psf), stf)
        @test eltype(yf) === TMCBF16 && all(isfinite, Float32.(yf))
        @test relerr(Float32.(yf), ref.y) < 5e-2
    end

    @testset "Float32 and Float64 data take the ordinary path" begin
        rng = Random.Xoshiro(23)
        core = TriMulCore(C, H; use_bias=true)
        ps, st = Lux.setup(rng, core)
        x = randn(rng, Float32, C, N, N, B)
        @test contraction_mode(x, st) isa FullPrecision
        @test contraction_mode(Float64.(x), st) isa FullPrecision
        y, _ = core(x, nothing, ps, st)
        @test eltype(y) === Float32
        @test eltype(first(core(Float64.(x), nothing, Lux.f64(ps), st))) === Float64
        # Folded parameters are valid in any precision: Float32 with the fold == Float32 without
        psf, stf = tmc_fold(ps, st, 3)
        @test first(core(x, nothing, psf, stf)) ≈ y rtol = 1e-4
    end

    @testset "batch independence in the 16-bit modes" begin
        rng = Random.Xoshiro(29)
        core = TriMulCore(C, H; use_bias=true, fused=false)
        ps0, st = Lux.setup(rng, core)
        ps = tmc_bump_values(tmc_perturb(rng, ps0), range(100.0f0, 400.0f0; length=H))
        x16 = Float16.(randn(rng, Float32, C, N, N, B))
        mask = rand(rng, Bool, N, N, B)
        psf, stf = tmc_fold(ps, st, 6)
        for (p, s) in ((tmc_cast(Float16, ps), st), (tmc_cast(Float16, psf), stf))
            y, _ = core(x16, mask, p, s)
            for b in 1:B
                yb, _ = core(x16[:, :, :, b:b], mask[:, :, b:b], p, s)
                @test relerr(y[:, :, :, b:b], yb) < 5e-3
            end
        end
    end

    @testset "type stability" begin
        rng = Random.Xoshiro(31)
        core = TriMulCore(C, H; use_bias=true)
        ps0, st = Lux.setup(rng, core)
        ps = tmc_perturb(rng, ps0)
        mask = rand(rng, Bool, N, N, B)
        x = randn(rng, Float32, C, N, N, B)
        psf, stf = tmc_fold(ps, st, 3)
        @test_nowarn @inferred core(x, mask, ps, st)                                         # FullPrecision
        @test_nowarn @inferred core(Float16.(x), mask, tmc_cast(Float16, ps), st)            # HalfSafe
        @test_nowarn @inferred core(Float16.(x), mask, tmc_cast(Float16, psf), stf)          # Folded
        @test_nowarn @inferred core(x, nothing, psf, stf)                                    # Folded, Float32
    end

    @testset "_apply_tri_mask" begin
        rng = Random.Xoshiro(37)
        for T in (Float32, Float16)
            ab = rand(rng, T, N * N * B, 2H)                     # token-major, tokens (i, j, b), i fastest
            ab0 = copy(ab)
            @test _apply_tri_mask(ab, nothing) === ab
            m3 = rand(rng, Bool, N, N, B)
            out3 = _apply_tri_mask(ab, m3)
            @test out3 == ab .* vec(m3) && eltype(out3) === T
            m2 = rand(rng, Bool, N, N)
            out2 = _apply_tri_mask(ab, m2)
            @test out2 == ab .* vec(repeat(m2, 1, 1, B))          # `[N, N]` is shared by every batch element
            @test _apply_tri_mask(ab, trues(N, N, B)) == ab
            @test all(iszero, _apply_tri_mask(ab, falses(N, N)))
            @test ab == ab0                                      # out of place
        end
    end
end
