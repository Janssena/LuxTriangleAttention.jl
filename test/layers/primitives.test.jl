# `StableLayerNorm` and `LayerNormNoBias` (src/layers/primitives.jl), and the Float32-statistics
# normalisation behind them (`layernorm_f32stats`, `_narrow`, `_normalize`). Pure Julia on CPU; the traced
# (Reactant) testset is `test/ext/reactant/primitives.test.jl`.
#
# The reference is always Float64 on the SAME 16-bit-rounded inputs and parameters: the wrapped
# `Lux.LayerNorm` run on Float64 data (the pass-through), or `LuxLib.layernorm` on Float64 (`layernorm_f64_ref`).

import LuxLib
using LuxTriangleAttention: layernorm_f32stats, _layernorm_f32stats, _narrow, _normalize,
    StableLayerNorm, LayerNormNoBias

const BF16 = LuxTriangleAttention.BFloat16

# LuxLib's `layernorm` on Float64 data: the reference every 16/32-bit result is held to. It has no
# scale-only or bias-only method, so the missing one is the identity.
function layernorm_f64_ref(x, scale, bias, act, dims, ε)
    x = Float64.(x)
    isnothing(scale) && isnothing(bias) && return LuxLib.layernorm(x, nothing, nothing, act, dims, Float64(ε))
    γ = isnothing(scale) ? ones(Float64, size(bias)) : Float64.(scale)
    β = isnothing(bias) ? zeros(Float64, size(γ)) : Float64.(bias)
    return LuxLib.layernorm(x, γ, β, act, dims, Float64(ε))
end

@testset "layernorm_f32stats" begin
    rng = Random.Xoshiro(3)
    ε = 1.0f-5

    @testset "overflow: Σx² beyond floatmax, T=$T" for T in (Float16, BF16)
        # σ = 30 over 128 channels: Σx² ≈ 1.2e5 > floatmax(Float16)
        x = T.(30 .* randn(rng, Float32, 128, 64) .+ 5)
        @test 128 * 30^2 > floatmax(Float16)
        ref = layernorm_f64_ref(x, nothing, nothing, identity, 1, ε)

        y = layernorm_f32stats(x, nothing, nothing, identity, 1, ε, T)
        @test eltype(y) === T && size(y) == size(x) && all(isfinite, y)
        @test y ≈ ref

        if T === Float16
            # Σx² → Inf, so x / Inf = 0: finite, but all information lost
            lux_ref = LuxLib.layernorm(x, nothing, nothing, identity, 1, T(ε))
            @test all(iszero, lux_ref)      
        end
    end

    @testset "large mean, small spread" begin
        x = Float16.(200 .+ 10 .* randn(rng, Float32, 64, 32))   # Σx² ≈ 2.6e6
        y = layernorm_f32stats(x, nothing, nothing, identity, 1, ε, Float16)
        @test all(isfinite, y)
        ref = layernorm_f64_ref(x, nothing, nothing, identity, 1, ε)
        @test y ≈ ref
    end

    @testset "dims = $dims" for dims in (1, 2, (1, 2), (1, 2, 3))
        x = Float16.(40 .* randn(rng, Float32, 16, 8, 4) .+ 3)
        y = layernorm_f32stats(x, nothing, nothing, identity, dims, ε, Float16)
        ref = layernorm_f64_ref(x, nothing, nothing, identity, dims, ε)
        @test all(isfinite, y)
        @test y ≈ ref
        # the normalised data has zero mean over `dims`
        @test maximum(abs, sum(Float64.(y); dims) ./ prod(d -> size(x, d), dims)) < 5e-2
    end

    @testset "scale, bias and activation" begin
        x = Float16.(30 .* randn(rng, Float32, 32, 16))
        scale = reshape(1.5f0 .+ rand(rng, Float32, 32), 32, 1)
        bias = reshape(randn(rng, Float32, 32), 32, 1)
        for (s, b) in ((scale, nothing), (nothing, bias), (scale, bias)), act in (identity, tanh, Lux.relu)
            y = layernorm_f32stats(x, s, b, act, 1, ε, Float16)
            @test eltype(y) === Float16                 # a Float32 scale/bias must not promote
            # reference on the Float16-rounded scale/bias, which is what the layer applies
            ref = layernorm_f64_ref(x, isnothing(s) ? nothing : Float16.(s), isnothing(b) ? nothing : Float16.(b), act, 1, ε)
            @test y ≈ ref
        end
        # scale and bias already in T are used as they are
        s16, b16 = Float16.(scale), Float16.(bias)
        @test layernorm_f32stats(x, s16, b16, identity, 1, ε, Float16) ==
              layernorm_f32stats(x, scale, bias, identity, 1, ε, Float16)
    end

    @testset "input type -> output type" begin
        x32 = 30 .* randn(rng, Float32, 32, 16)
        x16 = Float16.(x32)
        # wide data (a contraction computed wide) narrowed to T: normalised wide, then converted
        y = layernorm_f32stats(x32, nothing, nothing, identity, 1, ε, Float16)
        @test eltype(y) === Float16
        @test isapprox(y, layernorm_f64_ref(x32, nothing, nothing, identity, 1, ε); rtol=2e-3)
        # Float64 data too
        @test eltype(layernorm_f32stats(Float64.(x32), nothing, nothing, identity, 1, ε, Float16)) === Float16
        # Float32 in and out
        y32 = layernorm_f32stats(x32, nothing, nothing, identity, 1, ε, Float32)
        @test eltype(y32) === Float32
        @test isapprox(y32, layernorm_f64_ref(x32, nothing, nothing, identity, 1, ε); rtol=1e-5)
        # wide in, narrow out agrees with narrow in, narrow out up to the input's own rounding
        @test isapprox(layernorm_f32stats(x16, nothing, nothing, identity, 1, ε, Float16),
                       layernorm_f32stats(Float32.(x16), nothing, nothing, identity, 1, ε, Float16); rtol=2e-3)
    end

    @testset "epsilon below the data type's range" begin
        # ε = 1e-12 is 0 in Float16: it must be used in Float32 or a constant row gives 0/0. This is the
        # `Folded` TriMul case (ε·16⁻ᵏ), where X arrives wide.
        x = fill(7.0f0, 8, 4)
        y = layernorm_f32stats(x, nothing, nothing, identity, 1, 1.0f-12, Float16)
        @test eltype(y) === Float16 && all(iszero, y)
        xn = 7.0f0 .+ randn(rng, Float32, 8, 4)
        yn = layernorm_f32stats(xn, nothing, nothing, identity, 1, 1.0f-12, Float16)
        @test all(isfinite, yn) &&
              isapprox(yn, layernorm_f64_ref(xn, nothing, nothing, identity, 1, 1.0f-12); rtol=2e-3)
        # narrow data: ε far below Float16's smallest subnormal (6e-8) but 1/√ε still fits (< 65504)
        x16 = fill(Float16(7), 8, 4)
        @test all(iszero, layernorm_f32stats(x16, nothing, nothing, identity, 1, 1.0f-9, Float16))
    end

    @testset "type stability" begin
        x = Float16.(randn(rng, Float32, 8, 4))
        s = ones(Float32, 8, 1)
        @inferred layernorm_f32stats(x, nothing, nothing, identity, 1, ε, Float16)
        @inferred layernorm_f32stats(x, s, s, tanh, 1, ε, Float16)
        @inferred layernorm_f32stats(x, s, s, identity, (1, 2), ε, Float16)
        @inferred layernorm_f32stats(Float32.(x), s, nothing, identity, 1, ε, Float16)
        @test eltype(@inferred layernorm_f32stats(x, s, s, identity, 1, ε, Float16)) === Float16
    end
end

@testset "_narrow / _normalize" begin
    a16, a32 = Float16[1, 2, 3], Float32[1, 2, 3]
    @test _narrow(a16, Float16) === a16                  # already T: the same object, no copy
    @test _narrow(a32, Float16) isa Vector{Float16} && _narrow(a32, Float16) == a16
    @test _narrow(Float64[1, 2], Float32) isa Vector{Float32}
    @test @inferred(_narrow(a32, Float16)) isa Vector{Float16}

    μ, rstd = Float32[1.0 2.0], Float32[0.5 0.25]
    x16 = Float16[3 4; 5 6]
    y = _normalize(x16, μ, rstd, Float16)                # already in T
    @test eltype(y) === Float16 && y == Float16[1 0.5; 2 1]
    x32 = Float32[3 4; 5 6]
    y = _normalize(x32, μ, rstd, Float16)                # wide: normalised wide, then narrowed
    @test eltype(y) === Float16 && y == Float16[1 0.5; 2 1]
    @test @inferred(_normalize(x16, μ, rstd, Float16)) isa Matrix{Float16}
    @test @inferred(_normalize(x32, μ, rstd, Float16)) isa Matrix{Float16}
end

@testset "StableLayerNorm" begin
    LN = StableLayerNorm
    f64(x) = Float64.(x)
    # σ = 60 over 32 channels: Σx² ≈ 1.2e5 > floatmax(Float16)
    C, N, B, σ = 32, 6, 2, 60.0f0
    rng = Random.Xoshiro(11)
    x16 = Float16.(σ .* randn(rng, Float32, C, N, N, B) .+ 5.0f0)
    @test C * σ^2 > floatmax(Float16)

    @testset "construction" begin
        ln = LN((C, 1, 1); dims=1, epsilon=1.0f-5)
        @test ln.layer isa Lux.LayerNorm
        @test LN(ln.layer).layer === ln.layer
        @test LN((C, 1, 1), tanh).layer.activation === tanh
        @test occursin("StableLayerNorm(", sprint(show, ln))
        # LayerNormNoBias is a StableLayerNorm around a frozen layer
        nb = LayerNormNoBias((C, 1, 1); dims=1)
        @test nb isa LN && nb.layer isa Lux.Experimental.FrozenLayer
    end

    @testset "same parameters and state as Lux.LayerNorm" begin
        ref = Lux.LayerNorm((C, 1, 1); dims=1, epsilon=1.0f-5)
        ln = LN((C, 1, 1); dims=1, epsilon=1.0f-5)
        pr, sr = Lux.setup(Random.Xoshiro(1), ref)
        pl, sl = Lux.setup(Random.Xoshiro(1), ln)
        @test pl == pr && sl == sr                    # unnested: a checkpoint of either loads into the other
        @test LN(ref).layer === ref
    end

    @testset "$name" for (name, ln) in (
        ("StableLayerNorm", LN((C, 1, 1); dims=1, epsilon=1.0f-5)),
        ("LayerNormNoBias", LayerNormNoBias((C, 1, 1); dims=1, epsilon=1.0f-5)))
        ps, st = Lux.setup(rng, ln)
        # a non-trivial scale (and bias where the layer has one), so they are really applied
        ps = merge(ps, (; scale=ps.scale .* 1.5f0))
        haskey(ps, :bias) && (ps = merge(ps, (; bias=ps.bias .+ 0.25f0)))
        ps16, st16 = Lux.f16(ps), Lux.f16(st)
        ps64, st64 = Lux.f64(ps16), Lux.f64(st16)
        @test ln isa LN                                             # `LayerNormNoBias` builds one too

        y, st_out = ln(x16, ps16, st16)
        y64, _ = ln(f64(x16), ps64, st64)                           # Float64: the pass-through
        @test eltype(y) === Float16 && size(y) == size(x16)
        @test all(isfinite, y)
        @test isapprox(y, y64; rtol=5e-3)
        @test st_out == st16
        # the wrapped layer itself, on these Float16 numbers: the failure being fixed
        yl, _ = ln.layer(x16, ps16, st16)
        @test !all(isfinite, yl) || !(yl ≈ y64)

        # 32/64-bit data is the wrapped layer itself, bit for bit
        x32 = Float32.(x16)
        ps32, st32 = Lux.f32(ps), Lux.f32(st)
        @test first(ln(x32, ps32, st32)) == first(ln.layer(x32, ps32, st32))
        @test first(ln(f64(x16), ps64, st64)) == first(ln.layer(f64(x16), ps64, st64))
        @test eltype(first(ln(x32, ps32, st32))) === Float32
        @test eltype(first(ln(f64(x16), ps64, st64))) === Float64

        # parameters left in Float32 (e.g. a plain `Lux.f16(x)` of the data only): no promotion of Float16 data
        y_mixed, _ = ln(x16, ps, st)
        @test eltype(y_mixed) === Float16 && all(isfinite, y_mixed)
        @test isapprox(y_mixed, y64; rtol=5e-3)
    end

    @testset "BFloat16" begin
        ln = LN((C, 1, 1); dims=1, epsilon=1.0f-5)
        ps, st = Lux.setup(rng, ln)
        ps = merge(ps, (; scale=ps.scale .* 1.5f0, bias=ps.bias .+ 0.25f0))
        xb = BF16.(x16)
        pb = map(a -> BF16.(a), ps)
        y, _ = ln(xb, pb, st)
        y64, _ = ln.layer(f64(xb), Lux.f64(pb), st)
        @test eltype(y) === BF16 && all(isfinite, y)
        @test isapprox(y, y64; rtol=3e-2)                # BFloat16 has 8 significant bits
    end

    @testset "dims, shape and activation" begin
        # (dims, parameter shape) pairs over x::[C, N, N, B]
        cases = ((1, (C, 1, 1)), ((1, 2), (C, N, 1)), ((1, 2, 3), (C, N, N)))
        @testset "dims=$dims act=$act" for (dims, shape) in cases, act in (identity, tanh)
            ln = LN(shape, act; dims, epsilon=1.0f-5)
            ps, st = Lux.setup(rng, ln)
            ps = merge(ps, (; scale=ps.scale .* 1.5f0, bias=ps.bias .+ 0.25f0))
            ps16, st16 = Lux.f16(ps), Lux.f16(st)
            y, _ = ln(x16, ps16, st16)
            y64, _ = ln(f64(x16), Lux.f64(ps16), Lux.f64(st16))
            @test eltype(y) === Float16 && all(isfinite, y)
            @test isapprox(y, y64; rtol=5e-3)
        end

        @testset "affine = false" begin
            ln = LN((C, 1, 1); dims=1, affine=false)
            ps, st = Lux.setup(rng, ln)
            @test isempty(ps)
            y, _ = ln(x16, ps, st)
            y64, _ = ln(f64(x16), ps, st)
            @test eltype(y) === Float16 && all(isfinite, y) && isapprox(y, y64; rtol=5e-3)
        end
    end

    @testset "overflow regimes" begin
        ln = LN((C, 1, 1); dims=1, epsilon=1.0f-5)
        ps, st = Lux.setup(rng, ln)
        ps16, st16 = Lux.f16(ps), Lux.f16(st)
        ps64, st64 = Lux.f64(ps16), Lux.f64(st16)
        @testset "σ=$s μ=$m" for (s, m) in ((1f0, 0f0), (20f0, 0f0), (60f0, 5f0), (60f0, 1000f0), (500f0, 0f0))
            x = Float16.(s .* randn(rng, Float32, C, N, N, B) .+ m)
            all(isfinite, x) || continue
            y, _ = ln(x, ps16, st16)
            y64, _ = ln(f64(x), ps64, st64)
            @test all(isfinite, y)
            @test isapprox(y, y64; rtol=1e-2)
        end
        # an all-zero input (a fully padded position) is finite: 0 / √ε
        y, _ = ln(zeros(Float16, C, N, N, B), ps16, st16)
        @test all(isfinite, y) && all(==(0), y)
    end

    @testset "LayerNormNoBias keeps its zero bias in the state" begin
        ln = LayerNormNoBias((C, 1, 1); dims=1)
        ps, st = Lux.setup(rng, ln)
        @test ln isa LN && keys(ps) == (:scale,) && keys(st.frozen_params) == (:bias,)
        @test all(iszero, st.frozen_params.bias)
        # the tree a frozen Lux.LayerNorm has, unchanged by the wrapper
        fz = Lux.Experimental.freeze(Lux.LayerNorm((C, 1, 1); dims=1, affine=true, init_bias=Lux.zeros32), (:bias,))
        pf, sf = Lux.setup(Random.Xoshiro(1), fz)
        pl, sl = Lux.setup(Random.Xoshiro(1), ln)
        @test pl == pf && sl == sf
    end

    @testset "LayerNormNoBias = StableLayerNorm with a zero bias" begin
        nb = LayerNormNoBias((C, 1, 1); dims=1, epsilon=1.0f-5)
        ln = LN((C, 1, 1); dims=1, epsilon=1.0f-5)
        pn, sn = Lux.setup(rng, nb)
        pn = merge(pn, (; scale=pn.scale .* 1.5f0))
        pl = (; scale=pn.scale, bias=zero(pn.scale))
        _, sl = Lux.setup(rng, ln)
        for (T, cast) in ((Float16, Lux.f16), (Float32, Lux.f32))
            x = T.(x16)
            @test first(nb(x, cast(pn), cast(sn))) == first(ln(x, cast(pl), cast(sl)))
        end
        # the frozen bias kept in Float32 in the state does not widen the Float16 result
        y, _ = nb(x16, Lux.f16(pn), sn)
        @test eltype(y) === Float16 && all(isfinite, y)
    end

    @testset "_layernorm_f32stats on the wrapped layer" begin
        ln = LN((C, 1, 1); dims=1, epsilon=1.0f-5)
        ps, st = Lux.setup(rng, ln)
        ps = merge(ps, (; scale=ps.scale .* 1.5f0, bias=ps.bias .+ 0.25f0))
        ps16, st16 = Lux.f16(ps), Lux.f16(st)
        y, _ = ln(x16, ps16, st16)
        @test _layernorm_f32stats(ln.layer, x16, ps16) == y
        @test _layernorm_f32stats(ln.layer, x16, ps16, st16) == y
        # a frozen layer merges its frozen parameters back in
        nb = LayerNormNoBias((C, 1, 1); dims=1, epsilon=1.0f-5)
        pn, sn = Lux.setup(rng, nb)
        pn16, sn16 = Lux.f16(pn), Lux.f16(sn)
        @test _layernorm_f32stats(nb.layer, x16, pn16, sn16) == first(nb(x16, pn16, sn16))
    end

    @testset "type stability" begin
        ln = LN((C, 1, 1); dims=1, epsilon=1.0f-5)
        ps, st = Lux.setup(rng, ln)
        ps16, st16 = Lux.f16(ps), Lux.f16(st)
        @test eltype(first(@inferred ln(x16, ps16, st16))) === Float16
        @test eltype(first(@inferred ln(Float32.(x16), Lux.f32(ps), Lux.f32(st)))) === Float32
        nb = LayerNormNoBias((C, 1, 1); dims=1)
        pn, sn = Lux.setup(rng, nb)
        @test eltype(first(@inferred nb(x16, Lux.f16(pn), Lux.f16(sn)))) === Float16
        @test eltype(first(@inferred nb(x16, Lux.f16(pn), sn))) === Float16
    end

    @testset "ChannelsLast" begin
        CL = LuxTriangleAttention.ChannelsLast
        lnl = LN((C, 1, 1); dims=1)
        ps, st = Lux.setup(rng, lnl)
        ps = merge(ps, (; scale=ps.scale .* 1.5f0, bias=ps.bias .+ 0.25f0))
        xt = permutedims(x16, (2, 3, 4, 1))                           # channels last: [N, N, B, C]
        y, _ = CL(lnl)(xt, Lux.f16(ps), Lux.f16(st))
        y64, _ = CL(lnl.layer)(f64(xt), Lux.f64(Lux.f16(ps)), Lux.f64(Lux.f16(st)))
        @test eltype(y) === Float16 && all(isfinite, y) && isapprox(y, y64; rtol=5e-3)
        x32 = Float32.(xt)
        @test first(CL(lnl)(x32, Lux.f32(ps), Lux.f32(st))) == first(CL(lnl.layer)(x32, Lux.f32(ps), Lux.f32(st)))
        # and the channels-first layer agrees with the channels-last one
        yc, _ = lnl(x16, Lux.f16(ps), Lux.f16(st))
        @test isapprox(permutedims(yc, (2, 3, 4, 1)), y; rtol=5e-3)
    end
end
