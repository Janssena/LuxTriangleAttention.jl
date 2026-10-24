# The correctness gate's reference (`src/kernels/verify.jl`), with no device.

using Test, Random, LuxTriangleAttention
const LTAt = LuxTriangleAttention

@testset "verification reference" begin
    @testset "reference_backward is the gradient of reference_forward" begin
        rng = Xoshiro(3)
        dim, n, nouter, bh, mheads = 3, 5, 2, 2, 2    # one batch element, two heads: one mask
        q, k, v, dO = (randn(rng, dim, n, nouter, bh) for _ in 1:4)
        bias = randn(rng, n, n, bh)
        mask = falses(n, nouter, 1)
        mask[2, 1, 1] = true
        mask[4, 2, 1] = true
        loss(q, k, v, bias) = sum(LTAt.reference_forward(q, k, v, bias, mask, mheads)[1] .* dO)
        grads = LTAt.reference_backward(q, k, v, bias, mask, mheads, dO)
        h = 1e-6
        for (g, x, which) in zip(grads, (q, k, v, bias), 1:4)
            for idx in (1, length(x) ÷ 2, length(x))
                xp, xm = copy(x), copy(x)
                xp[idx] += h
                xm[idx] -= h
                args(y) = ntuple(i -> i == which ? y : (q, k, v, bias)[i], 4)
                fd = (loss(args(xp)...) - loss(args(xm)...)) / 2h
                @test g[idx] ≈ fd rtol = 1e-5 atol = 1e-8
            end
        end
    end

    @testset "a fully-masked row attends uniformly: o = mean(v), lse = -Inf" begin
        q, k, v = (randn(2, 3, 1, 1) for _ in 1:3)
        mask = trues(3, 1, 1)
        o, lse = LTAt.reference_forward(q, k, v, zeros(3, 3, 1), mask, 1)
        @test all(==(-Inf), lse)
        @test o ≈ repeat(sum(v; dims=2) ./ 3, 1, 3, 1, 1)
        # its gradient reaches v (dO / n per key) and nothing else
        dO = randn(2, 3, 1, 1)
        dq, dk, dv, dbias = LTAt.reference_backward(q, k, v, zeros(3, 3, 1), mask, 1, dO)
        @test all(iszero, dq) && all(iszero, dk) && all(iszero, dbias)
        @test dv ≈ repeat(sum(dO; dims=2) ./ 3, 1, 3, 1, 1)
    end

    @testset "the verification problem reaches every addressing path" begin
        V = LTAt.verification_problem(8, Float32)
        @test V.n % 2 == 1 && V.n != V.nouter            # ragged, rectangular
        @test V.bh ÷ V.mheads > 1                         # more than one batch mask
        @test size(V.mask, 3) == V.bh ÷ V.mheads
        @test V.mask[:, :, 1] != V.mask[:, :, 2]          # and they differ
        # exactly one dead slice: outer 2 of the first batch mask, live under the second
        @test count(i -> all(V.mask[:, i]), CartesianIndices(size(V.mask)[2:3])) == 1
        @test all(V.mask[:, 2, 1]) && !all(V.mask[:, 2, 2])
        @test count(==(-Inf), V.ref.lse) == V.n * V.mheads    # its queries, for the `mheads` batch·heads under that mask
        @test all(isfinite, V.ref.o)
    end

    @testset "verification_error" begin
        @test LTAt.verification_error([1.0, 2.0], [1.0, 2.0]) == 0
        @test LTAt.verification_error([1.0, NaN], [1.0, 2.0]) == Inf   # NaN never passes
        # a reference -Inf (a dead row's lse) must be matched exactly, and only it
        @test LTAt.verification_error([1.0, -Inf], [1.0, -Inf]) == 0
        @test LTAt.verification_error([1.0, 3.0], [1.0, -Inf]) == Inf
        @test LTAt.verification_error([1.0, -Inf], [1.0, 2.0]) == Inf
        @test LTAt.verification_error([1.0, NaN], [1.0, -Inf]) == Inf
        @test LTAt.verification_error([1.0, 2.2], [1.0, 2.0]) ≈ 0.1
    end
end
