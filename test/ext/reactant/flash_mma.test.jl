# `:flash_mma` under Reactant on the GPU's client (`traced_client()`): the `Attention` layer with
# `use_kernel!(:flash_mma)` traced into the backend's kernel custom call (`TRACED_CALL_TARGET`), against the same layer
# on the plain softmax path. The traced path binds the same explicit argument lists as the eager one
# (`KernelDispatch.forward`); what is new against `:default` is a binary that covers more rows per group
# than it has threads (its `tile`, which divides the launch's `extent`), and the untransposed bias.

using Test, Random, Reactant
import LuxTriangleAttention: Lux, Static.static

const _MPS_MMA = traced_client()

if _MPS_MMA === nothing || GPU === nothing
    @warn "no Reactant client for the GPU (or no functional GPU backend); skipping traced flash_mma tests" maxlog = 1
else
    const _PREV_BACKEND_MMA = Reactant.XLA.default_backend()
    Reactant.set_default_backend(_MPS_MMA)
    prev = LuxTriangleAttention.current_kernel()
    try
        @testset "traced Attention on :flash_mma, $T, head dimension $hd" for T in (Float32, Float16), hd in (16, 8)
            C, H, N, S, B = 32, 2, 24, 7, 1
            # A device that computes `T` at another type rounds its inputs to it (CUDA: Float32 at Float16), so the traced
            # kernel agrees with the softmax path to that type's accuracy, not Float32's.
            Tc = LuxTriangleAttention.mma_compute_type(LuxTriangleAttention.mma_backend(), T)
            isnothing(Tc) && (Tc = T)
            # A trace cannot tune; tune the length the layer will call with, ahead of time.
            tune_flash_mma!(N; dim=hd, T)
            tune_flash_kernel!(N; dim=hd, T)    # `:default` too: it is what a head dimension the matrix units cannot take falls back to
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
            @test rel < (Tc === Float32 ? 1e-5 : 1e-2)
            ir = repr(Reactant.@code_hlo optimize = false f(lk, dev(x), dev(bias), dev(mask), dev(ps), dev(st)))
            # A head dimension the matrix units cannot contract over (8 on NVIDIA) traces the `:default` kernel instead; either
            # way it is a kernel custom call, never the host path.
            @test occursin(TRACED_CALL_TARGET, ir)
            GPU === :metal && hd == 16 && @test occursin("lta_flash_mma_fwd", ir)   # the kernel's name rides in the call on Metal
        end
    finally
        use_kernel!(prev)
        Reactant.set_default_backend(_PREV_BACKEND_MMA)
    end
end
