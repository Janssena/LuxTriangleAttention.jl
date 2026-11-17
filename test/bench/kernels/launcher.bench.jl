# Eager forward + backward through the registry launcher, for `:default` and `:trifast`, at Boltz2's
# triangle-attention shape (`[32, 4, N, N, 1]`, pair bias, no mask). It times the WHOLE call —
# host-side role resolution, allocation, binding and the launches — which is what the KernelDispatch.jl
# extraction (docs/done/FEATURE_kernelops.md) touches; the kernels themselves are unchanged by it.
#
#     julia --project=test test/bench/kernels/launcher.bench.jl [out.csv]
#
# Run before and after the migration and compare. Per `metal_timing_noise`, differences under ~10%
# between two separate runs are noise; only a larger shift is a finding.

include(joinpath(@__DIR__, "registry_setup.jl"))   # BEFORE `using LuxTriangleAttention`
using LuxTriangleAttention, Metal, Random, Printf, Statistics
include(joinpath(@__DIR__, "..", "..", "support", "sdpa_bwd.jl"))   # sdpa_bwd

const LTA = LuxTriangleAttention

function time_call(f; reps=30)
    f(); Metal.synchronize()                       # warm up (compile, allocate)
    ts = Float64[]
    for _ in 1:reps
        t = time_ns(); f(); Metal.synchronize()
        push!(ts, (time_ns() - t) / 1e3)
    end
    return median(ts)
end

function main(out)
    tune_default_fixed!()
    register_trifast!()
    rng = Random.Xoshiro(0)
    rows = String["kernel,N,fwd_us,bwd_us"]
    for N in (128, 256, 512)
        q, k, v, dO = (MtlArray(randn(rng, Float32, 32, 4, N, N, 1)) for _ in 1:4)
        bias = MtlArray(randn(rng, Float32, N, N, 4, 1))
        for name in (:default, :trifast)
            kern = kernel_named(name)
            fwd() = KernelDispatch.forward(LTA.SDPA(), kern, q, k, v, nothing, bias)
            o, lse = fwd()
            tf = time_call(fwd)
            tb = time_call(() -> sdpa_bwd(q, k, v, nothing, bias, o, lse, dO; kernel=kern))
            @printf "%-8s N=%4d  fwd %9.1f µs  bwd %9.1f µs\n" name N tf tb
            push!(rows, "$name,$N,$(round(tf; digits=1)),$(round(tb; digits=1))")
        end
    end
    write(out, join(rows, "\n") * "\n")
    println("wrote ", out)
end

main(get(ARGS, 1, joinpath(@__DIR__, "results.launcher.before.csv")))
