# `TriMul` without a device (`src/kernels/trimul/`): the stage argument lists, pinned symbolically and against the
# compiled argument types; the host route; which calls the kernel declines, and why; the missing backward.
# The kernel's numbers are tested on the GPU (`trimul_op.gpu.test.jl`).

using Test, Random, LuxTriangleAttention
using BFloat16s: BFloat16
import LuxTriangleAttention: Lux
import KernelDispatch
const LTAtm = LuxTriangleAttention

const TM_ROLE_NAMES = (:xn, :x, :γi, :βi, :y, :X, :γo, :βo, :a, :b, :P, :bias, :mask, :out, :V, :G, :bo, :bg,
    :dout, :da, :db, :dx, :xin, :dy, :dy2, :γ, :part)
const TM_ROLES = NamedTuple{TM_ROLE_NAMES}(TM_ROLE_NAMES)
const TM_SCALARS = (; t0=:t0, n=:n, T=:T, C=:C, mmod=:mmod, εi=:εi, εo=:εo, ldx=:ldx, dyoff=:dyoff, ε=:ε)

const TM_WANT = Dict(
    :ln_in => (:xn, :x, :γi, :βi, :t0, :n, :C, :εi),
    :ln_out => (:y, :X, :γo, :βo, :t0, :n, :T, :εo),
    :glu => (:a, :b, :P, :bias, :mask, :t0, :n, :T, :mmod),
    :gate => (:out, :V, :G, :bo, :bg, :t0, :n),
    :gate_bwd => (:V, :G, :dout, :bo, :bg, :t0, :n),
    :glu_bwd => (:P, :da, :db, :bias, :mask, :t0, :n, :T, :mmod),
    :ln_out_bwd => (:dx, :xin, :dy, :dy2, :γ, :part, :t0, :n, :ldx, :dyoff, :ε),
    :ln_in_bwd => (:dx, :xin, :dy, :dy2, :γ, :part, :t0, :n, :ldx, :dyoff, :ε),
)

# A value of each argument's compiled type, by role: what the forward binds.
const TM_VALUES = merge(map(_ -> zeros(Float32, 1), TM_ROLES), (; mask=falses(1) |> collect))
const TM_SVALUES = (; t0=Int32(0), n=Int32(1), T=Int32(1), C=Int32(1), mmod=Int32(1), εi=1.0f-5, εo=1.0f-5,
    ldx=Int32(1), dyoff=Int32(0), ε=1.0f-5)

@testset "trimul_args" begin
    @test Set(keys(TM_WANT)) == Set((LTAtm.TRIMUL_STAGES..., LTAtm.TRIMUL_BWD_STAGES...))
    p = (; TB=32, NT=256, NTE=256)
    for (stage, want) in TM_WANT
        @testset ":$stage" begin
            @test LTAtm.trimul_args(Val(stage), TM_ROLES, TM_SCALARS) == want
            # The bound values have the compiled types, slot for slot (the `Val`s bind nothing).
            types = filter(T -> !(T <: Val), LTAtm.trimul_argtypes(stage, 64, 32, p, true, true))
            vals = LTAtm.trimul_args(Val(stage), TM_VALUES, TM_SVALUES)
            @test length(vals) == length(types)
            @test all(((v, T),) -> v isa T, zip(vals, types))
        end
    end
    # 16-bit storage: the buffers take the data's type, `a`, `b` and `X` that of the contraction.
    @testset "storage types" begin
        p = (; TB=32, NT=256, NTE=256)
        for (T, Ta) in ((Float16, Float16), (Float16, Float32), (BFloat16, Float32))
            vals16 = merge(map(_ -> zeros(T, 1), TM_ROLES), (; mask=falses(1), a=zeros(Ta, 1), b=zeros(Ta, 1), X=zeros(Ta, 1)))
            for stage in LTAtm.TRIMUL_STAGES
                types = filter(U -> !(U <: Val), LTAtm.trimul_argtypes(stage, 64, 32, p, true, true, T, Ta))
                vals = LTAtm.trimul_args(Val(stage), vals16, TM_SVALUES)
                @test all(((v, U),) -> v isa U, zip(vals, types))
            end
        end
    end
end

tm_problem(; kw...) = begin
    l = TriangleMultiplication(16, 12; kw...)
    ps, st = Lux.setup(Random.Xoshiro(3), l)
    x = randn(Random.Xoshiro(4), Float32, 16, 7, 7, 2)
    mask = rand(Random.Xoshiro(5), Bool, 7, 7, 2)
    l, ps, st, x, mask
end

@testset "TriMul on the host" begin
    l, ps, st, x, mask = tm_problem()
    lk = TriangleMultiplication(16, 12; use_kernel=true)
    @test l.use_kernel === LTAtm.static(false)
    @test lk.use_kernel === LTAtm.static(true)
    ref = first(l(x, mask, ps, st))
    # Host arrays run the layer's own forward, through the op or not.
    @test first(lk(x, mask, ps, st)) == ref
    @test first(TriMul()(l, x, mask, ps, st)) == ref
    @test_nowarn @inferred lk(x, mask, ps, st)
end

@testset "what the kernel declines" begin
    why(l, x, mask, ps, st) = LTAtm._trimul_unsupported(l, x, mask, ps, st)
    l, ps, st, x, mask = tm_problem()
    @test why(l, x, mask, ps, st) === nothing
    @test why(l, x, nothing, ps, st) === nothing
    @test why(l, x, mask[:, :, 1], ps, st) === nothing                  # a shared [N, N] mask
    @test occursin("parameters", why(l, Float16.(x), mask, ps, st))        # 16-bit data, Float32 parameters
    @test why(l, Float16.(x), mask, Lux.f16(ps), st) === nothing
    @test why(l, BFloat16.(x), mask, Lux.fmap(a -> a isa AbstractArray ? BFloat16.(a) : a, ps), st) === nothing
    @test occursin("Float32, Float16", why(l, Float64.(x), mask, Lux.f64(ps), st))
    @test occursin("parameters", why(l, x, mask, Lux.f64(ps), st))
    @test occursin("mask", why(l, x, Float32.(mask), ps, st))
    @test occursin("mask", why(l, x, mask[1:6, :, :], ps, st))
    psf, stf = LTAtm.fold_trimul(ps, st, 2)
    @test why(l, x, mask, psf, stf) === nothing                         # folded parameters run in the kernel
    # Gates other than the sigmoid.
    lt = TriangleMultiplication(16, 12)
    glu = lt.core.glu_ab.layer
    relu_glu = LTAtm.GatedLinearUnit(glu.fused, Lux.relu, glu.linear, glu.gate, glu.out_chn)
    core = lt.core
    lr = TriangleMultiplication(lt.layer_norm, LTAtm.TriMulCore(core.fused, core.is_outgoing,
        LTAtm.ChannelsLast(relu_glu), core.glu_a, core.glu_b, core.layer_norm_out, core.glu_out, core.chn_hidden))
    @test occursin("sigmoid", why(lr, x, mask, ps, st))
end

@testset "the backward declines what it does not cover" begin
    l, ps, st, x, mask = tm_problem()
    lk = TriangleMultiplication(16, 12; use_kernel=true)
    why(args...) = LTAtm._trimul_bwd_unsupported(args...)
    @test why(lk, x, mask, ps, st, x) === nothing
    @test occursin("Float32", why(lk, Float16.(x), mask, Lux.f16(ps), st, Float16.(x)))
    psf, stf = LTAtm.fold_trimul(ps, st, 2)
    @test occursin("fold", why(lk, x, mask, psf, stf, x))
    @test occursin("cotangent", why(lk, x, mask, ps, st, x[:, 1:3, :, :]))
    err = try
        KernelDispatch.backward(TriMul(), LTAtm.TriMulStaged(), (lk, Float16.(x), mask, Lux.f16(ps), st), nothing, (Float16.(x), nothing))
    catch e
        e
    end
    @test err isa ArgumentError && occursin("use_kernel=false", err.msg)
end
