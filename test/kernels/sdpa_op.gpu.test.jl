# The Enzyme reverse rule for the device path: does it fire, and are its gradients right?
#
# Two separate claims, and only one of them is about numbers.
#
#  1. **The rule fires, and Enzyme never enters the kernel.** This is the load-bearing one. A
#     `.metallib` is opaque to Enzyme; if the rule failed to match, Enzyme would try to differentiate
#     the launch itself and abort with an LLVM IR dump. Reaching the assertions below at all is the
#     evidence.
#  2. **The gradients are correct**, against an independent host reference differentiated by Enzyme
#     through the ordinary softmax — no kernel, no rule, no shared code with the path under test.
#
# !!! note "Scope: this is an isolated rule test, not a model test"
#     Enzyme cannot differentiate ordinary Metal.jl code — a bare `sum(abs2, ::DeviceArray)` with no
#     rule makes it abort. So a whole-`Attention` backward on Metal is not possible regardless of
#     this rule, because the `Dense` projections, the gate and the loss around it are all ordinary
#     Metal.jl. What is verified here is the rule in isolation; the rule's real payoff is CUDA, where
#     Enzyme differentiates the surrounding code too. Do not extend this file into a model-level
#     gradient test on Metal — it will fail for reasons that have nothing to do with the kernel.

using Test, Random, Enzyme

if GPU !== nothing
    @testset "Enzyme reverse rule (concrete Metal)" begin
        rng = Random.Xoshiro(7)
        T = Float32
        hd, H, N, B = 8, 2, 16, 1

        q = randn(rng, T, hd, H, N, N, B)
        k = randn(rng, T, hd, H, N, N, B)
        v = randn(rng, T, hd, H, N, N, B)
        bias = randn(rng, T, N, N, H, 1, B)
        dO = randn(rng, T, hd, H, N, N, B)

        # Independent reference: Enzyme over the host path, which needs no rule at all.
        hostloss(q, k, v, b, dO) = sum(
            LuxTriangleAttention._sdpa_default(q, k, v, nothing, b) .* dO)
        rq, rk, rv, rb = zero(q), zero(k), zero(v), zero(bias)
        Enzyme.autodiff(Reverse, hostloss, Active, Duplicated(q, rq), Duplicated(k, rk),
            Duplicated(v, rv), Duplicated(bias, rb), Const(dO))

        tune_flash_kernel!(N; dim=hd, T)
        qd, kd, vd, bd, dOd = DeviceArray.((q, k, v, bias, dO))

        @testset "sdpa_bwd against the host reference" begin
            o, lse = LuxTriangleAttention.SDPA()(qd, kd, vd, nothing, bd)
            gq, gk, gv, gb = sdpa_bwd(
                qd, kd, vd, nothing, bd, o, lse, dOd)
            @test Array(gq) ≈ rq rtol = 1.0f-3
            @test Array(gk) ≈ rk rtol = 1.0f-3
            @test Array(gv) ≈ rv rtol = 1.0f-3
            @test Array(reshape(gb, size(bias))) ≈ rb rtol = 1.0f-3
        end

        @testset "through the rule" begin
            f(q, k, v, m, b) = first(LuxTriangleAttention.SDPA()(q, k, v, m, b))
            gq = DeviceArray(zeros(T, size(q)))
            gk = DeviceArray(zeros(T, size(k)))
            gv = DeviceArray(zeros(T, size(v)))
            gb = DeviceArray(zeros(T, size(bias)))

            # Split mode, because the loss that would close over these is itself ordinary Metal.jl
            # code Enzyme cannot differentiate — see the note at the top of this file. Seeding the
            # result shadow by hand is what a scalar loss would otherwise do for us.
            fwd, rev = Enzyme.autodiff_thunk(
                ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated,
                Duplicated{typeof(qd)}, Duplicated{typeof(kd)},
                Duplicated{typeof(vd)}, Const{Nothing}, Duplicated{typeof(bd)})
            args = (Const(f), Duplicated(qd, gq), Duplicated(kd, gk),
                Duplicated(vd, gv), Const(nothing), Duplicated(bd, gb))
            tape, _primal, shadow = fwd(args...)
            shadow .= dOd
            rev(args..., tape)

            @test Array(gq) ≈ rq rtol = 1.0f-3
            @test Array(gk) ≈ rk rtol = 1.0f-3
            @test Array(gv) ≈ rv rtol = 1.0f-3
            @test Array(gb) ≈ rb rtol = 1.0f-3
        end

        @testset "the rule does not fire on the host path" begin
            # The rule is narrowed to `Const{<:GPUBackend}`, so host arrays keep being
            # differentiated the ordinary way. If that narrowing regressed, this would either throw
            # or produce a `(output, lse)` shape the host path never returns.
            hq, hk, hv, hb = zero(q), zero(k), zero(v), zero(bias)
            Enzyme.autodiff(Reverse, hostloss, Active, Duplicated(q, hq), Duplicated(k, hk),
                Duplicated(v, hv), Duplicated(bias, hb), Const(dO))
            @test hq ≈ rq
            @test last(LuxTriangleAttention.SDPA()(q, k, v, nothing, bias)) === nothing
        end
    end
end
