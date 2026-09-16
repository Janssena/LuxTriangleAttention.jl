# Cross-attention passes `x` as the `(q, k, v)` tuple rather than a single array. Everything the
# forward derives from `x` has to keep working in that form: `prep_bias` picks the bias layout from
# the query, the same way `_gate_maybe` gates on it. Regression guard — the tuple method for
# `prep_bias` was missing, so any biased cross-attention (AF3's `SequenceLocalAttentionPairBias`,
# all of Boltz-2) hit a `MethodError` while the self-attention path stayed green.

@testset "tuple (q, k, v) inputs" begin
    rng = Random.Xoshiro(42)
    T = Float32
    chn, hd, H, N, S, B = 32, 8, 4, 8, 6, 2

    @testset "prep_bias dispatches on q" begin
        # 4D query: [C, N, S, B], bias [H, Nq, Nk, B].
        q4 = randn(rng, T, chn, N, S, B)
        bias4 = randn(rng, T, H, N, N, B)
        for layout in (:kq, :qk)
            @test prep_bias(bias4, (q4, q4, q4), static(layout)) ==
                  prep_bias(bias4, q4, static(layout))
        end

        # 3D query: [C, N, B] — the "already in the right shape" path.
        q3 = randn(rng, T, chn, N, B)
        for layout in (:kq, :qk)
            @test prep_bias(bias4, (q3, q3, q3), static(layout)) ==
                  prep_bias(bias4, q3, static(layout))
        end

        # A 5D bias is passed through whether or not `x` is a tuple.
        bias5 = randn(rng, T, N, N, H, 1, B)
        @test prep_bias(bias5, (q4, q4, q4), static(:kq)) === bias5

        @test prep_bias(nothing, (q4, q4, q4), static(:kq)) === nothing
    end

    # A 3-tuple `(q, k, v)` needs the unfused `BranchLayer` projection; a 2-tuple `(q, kv)` goes
    # through the fused-kv `Chain`. Both feed `prep_bias` a tuple.
    @testset "Attention forward, biased cross-attention (q, k, v)" begin
        x = randn(rng, T, chn, N, S, B)
        bias = randn(rng, T, H, N, N, B)
        mask = rand(rng, Bool, N, S, B)

        l = Attention(chn, hd, H; bias_layout=:kq, fuse_qkv=false)
        ps, st = Lux.setup(rng, l)

        # Feeding the same array as q, k and v must reproduce the self-attention forward exactly:
        # the tuple only changes how the inputs are routed, not the maths.
        ref = first(l(x, bias, mask, ps, st))
        @test first(l((x, x, x), bias, mask, ps, st)) ≈ ref

        # And a genuine cross-attention (distinct k/v) runs and is finite.
        kv = randn(rng, T, chn, N, S, B)
        y = first(l((x, kv, kv), bias, mask, ps, st))
        @test size(y) == size(x)
        @test all(isfinite, y)

        # NamedTuple entry point, the way the downstream models call it.
        @test first(l((; x=(x, kv, kv), bias, mask), ps, st)) ≈ y
    end

    @testset "Attention forward, biased cross-attention (q, kv)" begin
        chn_kv = 16
        x = randn(rng, T, chn, N, S, B)
        kv = randn(rng, T, chn_kv, N, S, B)
        bias = randn(rng, T, H, N, N, B)
        mask = rand(rng, Bool, N, S, B)

        l = Attention(chn, chn_kv, chn_kv, hd, H; bias_layout=:kq)
        ps, st = Lux.setup(rng, l)

        y = first(l((x, kv), bias, mask, ps, st))
        @test size(y) == size(x)
        @test all(isfinite, y)
        @test first(l((; x=(x, kv), bias, mask), ps, st)) ≈ y
    end
end
