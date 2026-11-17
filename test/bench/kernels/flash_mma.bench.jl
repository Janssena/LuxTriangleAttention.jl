# `:flash_mma` (matrix units) against `:default` (the scalar flash kernel): GPU kernel time of the
# forward and the backward, through the registry and the real eager launchers.
#
#     julia --project=test test/bench/kernels/flash_mma.bench.jl            # every case, one process
#     julia --project=test test/bench/kernels/flash_mma.bench.jl 2 Float16  # one case (shape index, T)
#
# Appends one row per (case, pass) to `test/bench/kernels/results.flash_mma.csv` and prints a markdown
# table. Results and the reasoning behind this method: docs/done/FEATURE_flash_mma.md.
#
# ## Why GPU timestamps, not wall clock
#
# The eager launcher (KernelDispatch' Metal `execute`) is synchronous: it drains Metal.jl's queue,
# commits, and waits. That puts a fixed 3–7 ms on every call on an M1 Pro, which swamps everything
# at N ≤ 256. Here every launch a call makes is RECORDED (`KernelDispatch.record_bindings`, whose records
# carry each `Launch`), then replayed with `KernelDispatch.time_binary(launches)`: the call's launches in
# order, `reps` times in one command buffer, timed by its GPU timestamps. That is kernel time
# alone, for both kernels alike: `delta`, zeroing `dbias` and `:default`'s bias/dbias transposes are
# Metal.jl operations outside the recorded launches (and small: ~0.02 ms for the transpose at N=512).
#
# ## Pairing
#
# Single launches drift by several percent on Metal, so the two kernels are timed in interleaved
# rounds with alternating order and the speedup is the median of the per-round ratios, with its IQR.
#
# Run it on an idle GPU: another process on the GPU (a second Julia session, a training run) shows
# up as wide IQRs and inconsistent times — that is how it was first noticed.

using LuxTriangleAttention, Metal, Random, Printf, Statistics, Dates
include(joinpath(@__DIR__, "..", "..", "support", "sdpa_bwd.jl"))   # sdpa_bwd
import KernelDispatch
const LTA = LuxTriangleAttention

# The launches `f` makes, without running them. `f` runs with every launch replaced by a record, so
# its own results are meaningless: run it for real separately for those.
record(f) = [r.launch for r in KernelDispatch.record_bindings(f)]

"Mean GPU time of one replay of `launches`, in ms: all of them in order, `reps` times in one command buffer."
gputime(launches, reps) = KernelDispatch.time_binary(launches; reps, warmup=0) * 1e3

reps_for(recs) = clamp(ceil(Int, 60 / gputime(recs, 2)), 2, 400)    # ≈ 60 ms per sample

function paired(ra, rb; rounds=15)
    na, nb = reps_for(ra), reps_for(rb)
    ta, tb = Float64[], Float64[]
    for r in 1:rounds
        if isodd(r)
            push!(ta, gputime(ra, na)); push!(tb, gputime(rb, nb))
        else
            push!(tb, gputime(rb, nb)); push!(ta, gputime(ra, na))
        end
    end
    ratios = ta ./ tb
    return median(ta), median(tb), median(ratios), quantile(ratios, 0.25), quantile(ratios, 0.75)
end

relL2(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))

# (label, hd, H, N, Nj, B): triangle attention at Boltz2's head shape, and one MSA-row geometry.
const SHAPES = [
    ("tri N=128", 32, 4, 128, 128, 1),
    ("tri N=256", 32, 4, 256, 256, 1),
    ("tri N=384", 32, 4, 384, 384, 1),
    ("tri N=512", 32, 4, 512, 512, 1),
    ("msa-row N=256 S=128", 32, 8, 256, 128, 1),
]

const CSV = joinpath(@__DIR__, "results.flash_mma.csv")
const ONLY = length(ARGS) >= 2 ? (parse(Int, ARGS[1]), ARGS[2] == "Float16" ? Float16 : Float32) : nothing

rows = String[]
for (si, (label, hd, H, n, nouter, B)) in enumerate(SHAPES), T in (Float32, Float16)
    ONLY === nothing || (si, T) == ONLY || continue
    rng = Xoshiro(n)
    q, k, v, dO = (MtlArray(randn(rng, T, hd, H, n, nouter, B)) for _ in 1:4)
    bias = MtlArray(randn(rng, T, n, n, H, 1, B))
    mask = MtlArray(rand(rng, n, 1, 1, nouter, B) .> 0.1)
    up(x) = MtlArray(Float32.(Array(x)))

    tune_flash_kernel!(n; dim=hd, T)
    tune_flash_mma!(n; dim=hd, T)

    res = Dict{Symbol,Any}()
    # Each kernel called directly (`use_kernel!` only takes effect at the next top-level expression).
    for (kname, kern) in (:default => LTA.FlashScalar(), :flash_mma => LTA.FlashMMA())
        fwd() = KernelDispatch.forward(LTA.SDPA(), kern, q, k, v, mask, bias)
        fres = fwd()                                       # (o, lse), plus `e` with the `output_residual` preference
        o = fres[1]
        bwd() = KernelDispatch.backward(LTA.SDPA(), kern, (q, k, v, mask, bias), fres, (dO, nothing))
        rf, rb = record(fwd), record(bwd)
        g = bwd()
        res[kname] = (; rf, rb, o=Array(o), g=map(Array, (g[1], g[2], g[3], g[5])))
    end
    # Accuracy reference: `:default` at Float32 on the same values.
    tune_flash_kernel!(n; dim=hd, T=Float32)
    o32, l32 = LTA.SDPA()(up(q), up(k), up(v), mask, up(bias))
    g32 = map(Array, sdpa_bwd(up(q), up(k), up(v), mask, up(bias), o32, l32, up(dO)))
    o32 = Array(o32)

    for (pass, rk, flops) in (("fwd", :rf, 4.0), ("bwd", :rb, 10.0))
        fl = flops * n * n * nouter * H * B * hd           # fwd: QKᵀ, PV; bwd: 5 products
        ta, tb, sp, q25, q75 = paired(res[:default][rk], res[:flash_mma][rk])
        rd, rm = pass == "fwd" ?
            (relL2(res[:default].o, o32), relL2(res[:flash_mma].o, o32)) :
            (maximum(relL2(a, b) for (a, b) in zip(res[:default].g, g32)),
             maximum(relL2(a, b) for (a, b) in zip(res[:flash_mma].g, g32)))
        push!(rows, @sprintf("| %s | %s | %s | %.2f | %.2f | **%.2f×** [%.2f–%.2f] | %.2f | %.2f | %.1e | %.1e |",
            label, T, pass, ta, tb, sp, q25, q75, fl / ta / 1e9, fl / tb / 1e9, rd, rm))
        println(rows[end]); flush(stdout)
        isfile(CSV) || write(CSV, "timestamp,shape,T,pass,default_ms,flash_mma_ms,speedup,q25,q75,relL2_default,relL2_flash_mma\n")
        open(io -> println(io, join(Any[Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), "\"$label\"", T,
            pass, ta, tb, sp, q25, q75, rd, rm], ",")), CSV, "a")
    end
end

println("\n| shape | T | pass | :default (ms) | :flash_mma (ms) | speedup [IQR] | :default TFLOP/s | :flash_mma TFLOP/s | relL2 :default | relL2 :flash_mma |")
println("|---|---|---|---|---|---|---|---|---|---|")
foreach(println, rows)
