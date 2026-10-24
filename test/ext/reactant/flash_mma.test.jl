# `:flash_mma` under Reactant on the jax-mps Metal client: the `Attention` layer with
# `use_kernel!(:flash_mma)` traced into an `mps.metal_kernel_lib` custom call, against the same layer
# on the plain softmax path. The traced path binds the same explicit argument lists as the eager one
# (`KernelDispatch.forward`); what is new against `:default` is a binary that covers more rows per group
# than it has threads (its `tile`, which divides the launch's `extent`), and the untransposed bias.

using Test, Random, Reactant
import LuxTriangleAttention: Lux, Static.static

const _MPS_MMA = traced_client()

if _MPS_MMA === nothing || GPU === nothing
    @warn "no jax-mps Metal client (or Metal not functional); skipping traced flash_mma tests" maxlog = 1
else
    const _PREV_BACKEND_MMA = Reactant.XLA.default_backend()
    Reactant.set_default_backend(_MPS_MMA)
    prev = LuxTriangleAttention.current_kernel()
    try
        @testset "traced Attention on :flash_mma, $T" for T in (Float32, Float16)
            C, hd, H, N, S, B = 32, 16, 2, 24, 7, 1
            # A trace cannot tune; tune the length the layer will call with, ahead of time.
            tune_flash_mma!(N; dim=hd, T)
            use_kernel!(:flash_mma)
            rng = Random.Xoshiro(4)
            lk = Attention(C, hd, H; use_kernel=static(true))
            lh = Attention(C, hd, H; use_kernel=static(false))
            ps, st = Lux.setup(rng, lh)
            ps = Lux.fmap(x -> x isa AbstractArray{Float32} ? T.(x) : x, ps)
            x = randn(rng, T, C, N, S, B)
            bias = randn(rng, T, H, N, N, B)
            mask = fill(true, N, S, B)
            mask[2, 1, 1] = false
            dev = Reactant.to_rarray
            f(l, x, b, m, p, s) = first(l(x, b, m, p, s))
            yk = @jit f(lk, dev(x), dev(bias), dev(mask), dev(ps), dev(st))
            yh = @jit f(lh, dev(x), dev(bias), dev(mask), dev(ps), dev(st))
            @test eltype(yk) === T
            rel = sqrt(sum(abs2, Float64.(Array(yk)) .- Float64.(Array(yh))) / sum(abs2, Float64.(Array(yh))))
            @test rel < (T === Float32 ? 1e-5 : 1e-2)
            ir = repr(Reactant.@code_hlo optimize = false f(lk, dev(x), dev(bias), dev(mask), dev(ps), dev(st)))
            @test occursin("mps.metal_kernel_lib", ir) && occursin("lta_flash_mma_fwd", ir)
        end
    finally
        use_kernel!(prev)
        Reactant.set_default_backend(_PREV_BACKEND_MMA)
    end
end
