# The one place every built-in binary's argument list is written (`src/kernels/stage_args.jl`),
# pinned symbolically: every role and scalar bound to its own name, so a test failure reads as the
# list itself. No device. `docs/custom-kernels.md` §2 prints the same lists. The types and shapes they bind to are checked by
# KernelDispatch' signature check at every launch.

using Test, LuxTriangleAttention
const LTAsa = LuxTriangleAttention

const ROLES = NamedTuple{(:o, :lse, :q, :k, :v, :bias, :bias_t, :mask, :dO, :delta, :dq, :dk, :dv,
    :dbias, :dbias_t)}((:o, :lse, :q, :k, :v, :bias, :bias_t, :mask, :dO, :delta, :dq, :dk, :dv,
    :dbias, :dbias_t))
const SCALARS = (:scale, :n, :nouter, :nh, :mheads)

# `neg_inf` is a value, not a role: shown by name.
show_arg(x::Symbol) = x
show_arg(x::Float32) = x == LTAsa.NEG_INF_F32 ? :neg_inf : error("unexpected scalar $x")
listed(k, stage) = map(show_arg, LTAsa.stage_args(k, Val(stage), ROLES, SCALARS))

const WANT = Dict(
    (LTAsa.FlashScalar(), :fwd) => (:o, :lse, :q, :k, :v, :bias_t, :mask, :scale, :neg_inf, :n, :nouter, :nh, :mheads),
    (LTAsa.FlashScalar(), :dq) => (:dq, :dbias_t, :q, :k, :v, :bias_t, :mask, :dO, :lse, :delta, SCALARS...),
    (LTAsa.FlashScalar(), :dk) => (:dk, :q, :k, :v, :bias, :mask, :dO, :lse, :delta, SCALARS...),
    (LTAsa.FlashScalar(), :dv) => (:dv, :q, :k, :bias, :mask, :dO, :lse, SCALARS...),
    (LTAsa.FlashMMA(), :fwd) => (:o, :lse, :q, :k, :v, :bias, :mask, SCALARS...),
    (LTAsa.FlashMMA(), :dkv) => (:dk, :dv, :dbias, :q, :k, :v, :bias, :mask, :dO, :lse, :delta, SCALARS...),
    (LTAsa.FlashMMA(), :dq) => (:dq, :q, :k, :v, :bias, :mask, :dO, :lse, :delta, SCALARS...),
)

@testset "stage_args" begin
    for ((k, stage), want) in WANT
        @testset "$(nameof(typeof(k))) :$stage" begin
            @test listed(k, stage) == want
            # The outputs lead, in the order the binary returns them.
            outs = LTAsa.stage_outputs(k, stage)
            @test want[1:length(outs)] == outs
        end
    end
    # Every stage of both kernels is covered.
    @test Set(s for ((k, s), _) in WANT if k isa LTAsa.FlashScalar) == Set(LTAsa.FLASH_STAGES)
    @test Set(s for ((k, s), _) in WANT if k isa LTAsa.FlashMMA) == Set(LTAsa.FLASH_MMA_STAGES)
end
