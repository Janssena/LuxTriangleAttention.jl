# The traced :default forward is wrong on CUDA (test/scratch/cuda_traced_spike.jl: relative error 0.84 in every slice).
# HOW is it wrong? For a few small problems this prints, traced against the SAME kernel launched eagerly:
#   - the fraction of entries that agree (1e-4), and whether the traced values are a PERMUTATION of the right ones
#     (equal after sorting: a layout / stride problem) or different numbers altogether;
#   - the same for `lse`;
#   - a handful of values, and whether the output is all zeros (the kernel never ran / wrote nothing) or looks like `v`.
#
#   KERNELDISPATCH_CACHE=$(mktemp -d) julia --project=test test/scratch/cuda_traced_spike2.jl 2>&1 | tee lta_cuda_traced_spike2.log

using CUDA, Reactant, KernelDispatch, LuxTriangleAttention, Random, Printf, Statistics

const LTA = LuxTriangleAttention
Reactant.set_default_backend("gpu")

relerr(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / max(sum(abs2, Float64.(b)), eps()))
agree(a, b) = mean(isapprox.(a, b; rtol=1e-4, atol=1e-6))

function run_case(; hd, H, N, hb, hm, B=1, T=Float32)
    rng = Random.Xoshiro(7)
    q, k, v = (randn(rng, T, hd, H, N, N, B) for _ in 1:3)
    bias = hb ? randn(rng, T, N, N, H, 1, B) : nothing
    mask = hm ? (m = fill(true, N, 1, 1, N, B); m[2, :, :, 3, :] .= false; m) : nothing
    LTA.tune_flash_kernel!(N; dim=hd, T)
    dev = Reactant.to_rarray
    d(x) = x === nothing ? nothing : dev(x)
    c(x) = x === nothing ? nothing : CuArray(x)
    eo, el = LTA.SDPA()(c(q), c(k), c(v), c(mask), c(bias))
    f(q, k, v, m, b) = LTA.SDPA()(q, k, v, m, b)
    to, tl = Reactant.@jit f(d(q), d(k), d(v), d(mask), d(bias))
    eo, el, to, tl = Array(eo), Array(el), Array(to), Array(tl)
    @printf("hd=%-2d H=%d N=%-2d bias=%-5s mask=%-5s | out: err %.2e agree %.3f permutation %.1e | lse: err %.2e agree %.3f | zeros? %s | traced≈v? %.2e\n",
        hd, H, N, hb, hm, relerr(to, eo), agree(to, eo), relerr(sort(vec(to)), sort(vec(eo))),
        relerr(tl, el), agree(tl, el), all(iszero, to), relerr(to, v))
    return (; eo, to, el, tl)
end

println("--- bias / mask, head dimension 8, 4 heads")
for hb in (false, true), hm in (false, true)
    run_case(; hd=8, H=4, N=8, hb, hm)
end
println("--- head dimension and head count (bias + mask)")
for (hd, H) in ((8, 1), (16, 1), (16, 4), (32, 4))
    run_case(; hd, H, N=8, hb=true, hm=true)
end
println("--- sequence length")
for N in (4, 16, 37)
    run_case(; hd=8, H=2, N, hb=true, hm=true)
end

println("--- first values, hd=8 H=1 N=8, no bias, no mask")
r = run_case(; hd=8, H=1, N=8, hb=false, hm=false)
println("    eager : ", round.(vec(r.eo)[1:8]; digits=4))
println("    traced: ", round.(vec(r.to)[1:8]; digits=4))
println("    lse eager : ", round.(vec(r.el)[1:6]; digits=4))
println("    lse traced: ", round.(vec(r.tl)[1:6]; digits=4))
