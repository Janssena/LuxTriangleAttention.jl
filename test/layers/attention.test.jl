# 16-bit data through `src/layers/attention.jl`'s `Attention`, against a Float64 reference on the SAME
# Float16-rounded inputs and parameters, eager on CPU (the compiled cases are `test/ext/reactant/attention.test.jl`). The eager Float16 case
# cannot see the logit overflow — eagerly LuxLib scales q by 1/√d BEFORE Q·Kᵀ, but the compiler moves
# that multiply after the matmul (legal in exact arithmetic), so the unscaled logits must fit Float16.
# Reproduces the BioFold scratch `sdpa_scale_repro.jl` (Boltz2: unscaled logits 8.9e4). The chunked
# (`chunk_size`) path is in `test/ext/reactant/chunk.test.jl`; the fused kernels in `test/kernels/`.
#
# Written before the fixes: a failure is a finding. Each testset says what a failure would mean.

using Enzyme

@testset "Attention, 16-bit" begin
    @testset "logits beyond Float16 before the 1/√d scale" begin
        c = f16_overflow_attention(Random.Xoshiro(1))
        # The case is only meaningful inside this window.
        @test c.maxlogit > floatmax(Float16)
        @test c.maxscaled < floatmax(Float16) / 1.5
        y64, _ = c.layer(c.x64, c.ps64, c.st64)

        @testset "eager" begin
            y16, _ = c.layer(c.x16, c.ps16, c.st16)
            @test eltype(y16) === Float16
            @test all(isfinite, y16)
            @test relerr(y16, y64) < 2.0e-2
        end
    end

    @testset "chunked CPU, logits beyond Float16 before the scale" begin
        # A pair input, so `chunk_size` really slices (dim 4 is `Nj`); chunk sizes that divide N = 8, that do
        # not, and the degenerate ones. Same answer as unchunked and as the Float64 reference.
        c = f16_overflow_attention(Random.Xoshiro(1); pair=true)
        @test c.maxlogit > floatmax(Float16) && c.maxscaled < floatmax(Float16) / 1.5
        y64, _ = c.layer(c.x64, c.ps64, c.st64)
        y_ref, _ = c.layer(c.x16, c.ps16, c.st16)
        for cs in (1, 3, 4, 8, 11)
            l = Attention(16, 16, 2; chunk_size=cs)
            y16, _ = l(c.x16, c.ps16, c.st16)
            @test all(isfinite, y16)
            @test relerr(y16, y64) < 2.0e-2
            @test y16 == y_ref                       # chunking changes nothing
        end
    end

    @testset "`_sdpa_default`: a fully masked key set attends uniformly" begin
        # Defined behaviour (OpenFold/Boltz2's finite `-1e9` mask gives it): o = mean of v over the keys,
        # dV = dO / n per key, and no gradient to q, k or bias. [hd, H, Nq, Nj, B], mask [Nk, 1, 1, Nj, B].
        rng = Random.Xoshiro(9)
        hd, H, N, B = 8, 2, 6, 2
        q, k, v, dO = (randn(rng, hd, H, N, N, B) for _ in 1:4)
        bias = randn(rng, N, N, H, 1, B)
        mask = fill(true, N, 1, 1, N, B); mask[2, :, :, 3, :] .= false      # a sparse hole
        mask[:, :, :, 5, 1] .= false                                         # a dead slice: Nj = 5, batch 1
        dead(j, b) = j == 5 && b == 1

        o = LuxTriangleAttention._sdpa_default(q, k, v, mask, bias)
        @test all(isfinite, o)
        for j in 1:N, b in 1:B
            dead(j, b) && @test o[:, :, :, j, b] ≈ repeat(sum(v[:, :, :, j, b]; dims=3) ./ N, 1, 1, N)
        end
        # the live slices are untouched by the rule
        o_live = LuxTriangleAttention._lux_sdpa(q, k, v, mask, bias)
        @test o[:, :, :, 3, 1] ≈ o_live[:, :, :, 3, 1] && o[:, :, :, 5, 2] ≈ o_live[:, :, :, 5, 2]
        o_ref, _ = reference_sdpa(q, k, v, mask, bias)
        @test o ≈ o_ref

        loss(q, k, v, bias, mask, dO) = sum(LuxTriangleAttention._sdpa_default(q, k, v, mask, bias) .* dO)
        dq, dk, dv, db = zero(q), zero(k), zero(v), zero(bias)
        Enzyme.autodiff(Reverse, loss, Active, Duplicated(q, dq), Duplicated(k, dk),
            Duplicated(v, dv), Duplicated(bias, db), Const(mask), Const(dO))
        @test all(isfinite, dq) && all(isfinite, dk) && all(isfinite, dv) && all(isfinite, db)
        @test all(iszero, dq[:, :, :, 5, 1]) && all(iszero, dk[:, :, :, 5, 1])
        @test dv[:, :, :, 5, 1] ≈ repeat(sum(dO[:, :, :, 5, 1]; dims=3) ./ N, 1, 1, N)
    end

    @testset "fully masked key set is finite, $T" for T in (Float32, Float16)
        # Pair input [C, Ni, Nj, B], mask [Ni, Nj, B]: column j = 3 masks every key. Today the masked
        # logits are -Inf, so the softmax is NaN there.
        rng = Random.Xoshiro(2)
        chn, N, B = 16, 8, 1
        l = Attention(chn, 8, 2; bias_layout=:qk)
        ps, st = Lux.setup(rng, l) |> convert_types(T)
        x = T.(randn(rng, Float32, chn, N, N, B))
        mask = fill(true, N, N, B); mask[:, 3, :] .= false
        ok = setdiff(1:N, 3)

        y_eager, _ = l(x, nothing, mask, ps, st)
        @test all(isfinite, y_eager)
    end
end

# ───────────────────────────────────────────────────────────────────────────────────────────────────
# Layer routing: `use_kernel` / `chunk_size` dispatch (this was `attention.routing.jl`)

# `Attention`/`TriangleAttention`'s own dispatch on `chunk_size` and `use_kernel`: each keeps the
# forward output unchanged relative to neither being set, and the two are rejected together at
# construction (a fused kernel already tiles the keys, so there is nothing left for `chunk_size` to
# slice). SDPA-path-level routing (does `SDPA()` itself return the right thing) is
# `test/kernels/sdpa_op.test.jl`; this is the layer-level seam that picks a path.

@testset "use_kernel / chunk_size routing" begin
    rng = Random.Xoshiro(42)
    T = Float32
    hd, H, N, B = 8, 4, 8, 1

    x = randn(rng, T, 32, N, N, B)
    mk(; kw...) = TriangleAttention(32, hd, H; is_starting=true, kw...)
    base = mk()
    ps, st = Lux.setup(rng, base)
    ref = first(base(x, ps, st))

    @test first(mk(; chunk_size=4)(x, ps, st)) ≈ ref
    @test first(mk(; use_kernel=static(true))(x, ps, st)) ≈ ref rtol = 1.0f-5

    # `chunk_size` and `use_kernel` are alternatives: a kernel already tiles the keys, so
    # there is no full score tensor left to slice. Rejected at construction.
    @test_throws Exception mk(; chunk_size=4, use_kernel=static(true))
end

# ───────────────────────────────────────────────────────────────────────────────────────────────────
# Cross-attention `(q, k, v)` tuple inputs (this was `attention.tuple_inputs.jl`)

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

        # The 4-argument form `Attention(chn_q, chn_kv, head_dim, num_heads)` is the same layer
        # (it referenced an undefined `chn_in` and threw `UndefVarError`).
        l4 = Attention(chn, chn_kv, hd, H; bias_layout=:kq)
        @test l4 isa Attention
        @test Lux.setup(Random.Xoshiro(1), l4) == Lux.setup(Random.Xoshiro(1), l)
        @test first(l4((x, kv), bias, mask, ps, st)) ≈ y
    end
end
