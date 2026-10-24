# `GatedLinearUnit` and `SwiGLU` (src/layers/glu.jl) on channels-first data, against the formula
# `y = (Wᵥx + bᵥ) .* act.(W_g x + b_g)`. The fused layer holds both projections in one Dense, values first and
# gates after (`[2·out, in]`); the unfused one has a Dense each and, alone, takes two inputs. `ChannelsLast` over a
# GLU is in `channels_last.test.jl`.

@testset "GatedLinearUnit" begin
    rng = Random.Xoshiro(5)
    C, O = 6, 5
    x = randn(rng, Float32, C, 3, 4)
    dense(W, b, x) = reshape(W * reshape(x, size(x, 1), :), size(W, 1), size(x)[2:end]...) .+ b

    @testset "fused=$fused $(act === Lux.swish ? "SwiGLU" : "GLU") bias=$use_bias" for fused in (true, false),
            act in (Lux.sigmoid, Lux.swish), use_bias in (true, false)

        g = GatedLinearUnit(C => O; fused, activation=act, use_bias)
        ps, st = Lux.setup(rng, g)
        y, _ = g(x, ps, st)
        @test size(y) == (O, 3, 4)

        Wv, Wg, bv, bg = if fused
            W = ps.linear.weight
            b = use_bias ? ps.linear.bias : zeros(Float32, 2O)
            W[1:O, :], W[O+1:end, :], b[1:O], b[O+1:end]
        else
            ps.linear.weight, ps.gate.weight,
            use_bias ? ps.linear.bias : zeros(Float32, O), use_bias ? ps.gate.bias : zeros(Float32, O)
        end
        bcast(b) = reshape(b, :, 1, 1)
        @test y ≈ dense(Wv, bcast(bv), x) .* act.(dense(Wg, bcast(bg), x))
        @test_nowarn @inferred g(x, ps, st)
    end

    @testset "SwiGLU is a GLU with swish" begin
        s = SwiGLU(C => O; fused=false)
        g = GatedLinearUnit(C => O; fused=false, activation=Lux.swish)
        ps, st = Lux.setup(rng, s)
        @test first(s(x, ps, st)) == first(g(x, ps, st))
    end

    @testset "fused and unfused agree when the weights are split" begin
        gf = GatedLinearUnit(C => O; fused=true, use_bias=true)
        gu = GatedLinearUnit(C => O; fused=false, use_bias=true)
        psf, stf = Lux.setup(rng, gf)
        _, stu = Lux.setup(rng, gu)
        W, b = psf.linear.weight, psf.linear.bias
        psu = (linear=(weight=W[1:O, :], bias=b[1:O]), gate=(weight=W[O+1:end, :], bias=b[O+1:end]))
        @test first(gu(x, psu, stu)) ≈ first(gf(x, psf, stf))
    end

    @testset "two inputs (unfused only)" begin
        g = GatedLinearUnit((4, C) => O; fused=false)
        ps, st = Lux.setup(rng, g)
        v_in = randn(rng, Float32, 4, 3, 4)
        y, _ = g((v_in, x), ps, st)
        @test y ≈ dense(ps.linear.weight, 0, v_in) .* Lux.sigmoid.(dense(ps.gate.weight, 0, x))
        gf = GatedLinearUnit(C => O; fused=true)
        psf, stf = Lux.setup(rng, gf)
        @test_throws ErrorException gf((x, x), psf, stf)
    end

    @testset "fused needs one input width" begin
        @test_throws AssertionError GatedLinearUnit((4, C) => O; fused=true)
    end

    @testset "eltype preserved: $T" for T in (Float16, Float64), fused in (true, false)
        g = GatedLinearUnit(C => O; fused, use_bias=true)
        ps, st = Lux.setup(rng, g) |> convert_types(T)
        @test eltype(first(g(T.(x), ps, st))) === T
    end
end
