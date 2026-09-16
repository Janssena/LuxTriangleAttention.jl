# The ordinary `Attention` layer on the fused kernel (`use_kernel=true`), against the SAME layer and
# parameters on the plain softmax path (`use_kernel=false`), both traced on the jax-mps Metal client.
#
# Triangle attention is the one shape the kernel was written for (`Nq == Nj`) and is covered by
# `kernel_launch.test.jl`. This file is everything else the layer is used for:
#
#   * a plain sequence `[C, N, B]` — rank-4 q, reshaped to an outer extent of 1 (`_seq5`);
#   * MSA row attention `[C, N, S, B]` with a `:kq` pair bias `[H, N, N, B]` — attends over `N` for
#     each of `S` rows, `S` both above and below `N` so neither extent can stand in for the other;
#   * column-style attention with NO bias — the launcher substitutes a zero bias.
#
# Each case also asserts the kernel's custom_call is in the IR: the host path is numerically
# identical, so a quiet fallback would pass every comparison while testing nothing.

using Test, Random, Reactant, Metal
import LuxTriangleAttention: Lux, Static.static
# Guarded: re-including would reset `mps_client`'s cache, and a SECOND client for the same device
# fails to load — which returned `nothing` and silently skipped this whole file.
isdefined(@__MODULE__, :mps_client) || include(joinpath(@__DIR__, "..", "..", "mps_client.jl"))

const _MPS_ATT = mps_client()

# One sparse hole per outer slice, never a whole key row: a fully-masked row softmaxes to NaN on
# both paths, which would compare equal and hide everything else.
function _holes(dims...)
    m = fill(true, dims...)
    m[2, ntuple(Returns(1), length(dims) - 1)...] = false
    m[end, ntuple(i -> dims[i+1], length(dims) - 1)...] = false
    return m
end

if _MPS_ATT === nothing
    @warn "jax-mps Metal PJRT plugin not found; skipping Attention-on-kernel tests" maxlog = 1
elseif !Metal.functional()
    @warn "Metal not functional; skipping Attention-on-kernel tests" maxlog = 1
else
    const _PREV_BACKEND_ATT = Reactant.XLA.default_backend()
    Reactant.set_default_backend(_MPS_ATT)
    try
        @testset "Attention(use_kernel=true) == use_kernel=false" begin
            T = Float32
            C, B = 24, 1
            cases = [
                # (name, head_dim, heads, x dims, bias?, mask?)
                ("sequence [C,N,B], bias + mask", 8, 4, (C, 13, B), true, true),
                ("sequence [C,N,B], no bias", 16, 2, (C, 13, B), false, true),
                ("MSA row [C,N,S,B], S > N, bias + mask", 8, 4, (C, 7, 11, B), true, true),
                ("MSA row [C,N,S,B], S < N, bias + mask", 32, 2, (C, 12, 5, B), true, true),
                ("column-style [C,S,N,B], no bias, mask", 8, 4, (C, 9, 6, B), false, true),
                ("MSA row, no mask", 16, 2, (C, 10, 3, B), true, false),
            ]
            for (name, hd, H, xdims, with_bias, with_mask) in cases
                @testset "$name" begin
                    rng = Random.Xoshiro(11)
                    N = xdims[2]                     # the attended extent
                    LuxTriangleAttention.tune_flash_kernel!(N; dim=hd, T)

                    lk = Attention(C, hd, H; use_kernel=static(true))
                    lh = Attention(C, hd, H; use_kernel=static(false))
                    ps, st = Lux.setup(rng, lh)

                    x = randn(rng, T, xdims...)
                    bias = with_bias ? randn(rng, T, H, N, N, B) : nothing
                    mask = with_mask ? _holes(xdims[2:end]...) : nothing

                    dev = Reactant.to_rarray
                    xd, psd, std = dev(x), dev(ps), dev(st)
                    bd = bias === nothing ? nothing : dev(bias)
                    md = mask === nothing ? nothing : dev(mask)
                    f(l, x, b, m, p, s) = first(l(x, b, m, p, s))

                    yk = Array(@jit f(lk, xd, bd, md, psd, std))
                    yh = Array(@jit f(lh, xd, bd, md, psd, std))
                    y_host = first(lh(x, bias, mask, ps, st))   # plain Julia, CPU

                    @test size(yk) == size(x)
                    @test yk ≈ yh rtol = 1.0f-5 atol = 1.0f-6
                    @test yk ≈ y_host rtol = 1.0f-5 atol = 1.0f-6

                    ir = repr(Reactant.@code_hlo optimize = false f(lk, xd, bd, md, psd, std))
                    @test occursin("stablehlo.custom_call", ir)
                end
            end
        end

        @testset "rectangular on a pre-`nouter` stage is refused, not mis-addressed" begin
            # A stage registered before the argument existed declares no `:nouter`, and its binary
            # hard-codes the outer extent to `n`. It must still run square problems and must
            # REFUSE rectangular ones — silently it would walk the wrong rows.
            old = LuxTriangleAttention.KernelStage(:fwd, "f", "e", "", [:o, :lse],
                [:q, :k, :v, :bias, :mask], [:scale, :neg_inf, :n, :nh, :mheads],
                (:ka_ngroups, :one, :one), 32)
            entry = (; name=:default)
            @test LuxTriangleAttention.check_nouter(entry, (old,), 8, 8) === nothing
            @test_throws ArgumentError LuxTriangleAttention.check_nouter(entry, (old,), 8, 5)
        end
    finally
        Reactant.set_default_backend(_PREV_BACKEND_ATT)
    end
end
