# Backward accuracy of `:flash_mma`, as a multiple of the storage type's rounding floor.
#
#     KERNELDISPATCH_CACHE=$(mktemp -d) julia --project=test test/bench/kernels/accuracy.bench.jl
#     LTA_BENCH_GPU=cuda …      to pick the GPU (default: Metal on a Mac, CUDA elsewhere)
#
# The protocol is the one in TurboPairFormer (arXiv:2610.05854, §3.2 and App. G.3): q, k, v standard normal with, with
# probability 0.001, an extra normal outlier of standard deviation 10; dO standard normal; every operand rounded ONCE to the
# storage type; the reference is the Float64 gradient of the same rounded values (Enzyme through `_sdpa_default`, as in
# `test/kernels/flash_mma/runtests.jl`). The error is the absolute RMSE of each public gradient; the FLOOR is the RMSE of the
# Float64 reference rounded once to the storage type, the smallest error any tensor of that type can have, so `ratio = 1` is
# perfect. Their kernel without / with the output residual: dQ 2.8-3.4x / 1.8x, dK and dBias 2.0-2.5x / 1.4x (BFloat16).
#
# Two settings per shape: "plain" (no bias, no mask: the paper's) and "bias+mask" (what the layers pass). `dv` is the
# control: the residual and the dbias rework must leave it unchanged.
#
# Appends one row per (device, type, shape, setting, gradient) to `test/bench/kernels/results.accuracy.csv`; compare two
# builds by running this before and after a change on the same device.

using LuxTriangleAttention, KernelDispatch, Random, Printf, Statistics, Dates, Enzyme

const LTA = LuxTriangleAttention
const KD = KernelDispatch

const GPU = lowercase(get(ENV, "LTA_BENCH_GPU", Sys.isapple() ? "metal" : "cuda"))
if GPU == "metal"
    using Metal
    const DEV = Metal.MtlArray
    const DEVNAME = "Metal"
else
    using CUDA
    const DEV = CUDA.CuArray
    const DEVNAME = CUDA.name(CUDA.device())
end

# (hd, H, n): Pairformer head shape; kept small enough for the Float64 Enzyme reference on the host.
const SHAPES = [(32, 4, 96), (32, 4, 192), (16, 4, 192)]
const TYPES = (Float16, LTA.BFloat16)
const SEEDS = 3
const CSV = joinpath(@__DIR__, "results.accuracy.csv")

rmse(a, b) = sqrt(mean(abs2, Float64.(a) .- Float64.(b)))

function operands(rng, hd, H, n)
    r(dims...) = randn(rng, Float64, dims...) .+ (rand(rng, dims...) .< 0.001) .* (10 .* randn(rng, Float64, dims...))
    return r(hd, H, n, n, 1), r(hd, H, n, n, 1), r(hd, H, n, n, 1), randn(rng, Float64, hd, H, n, n, 1)
end

function reference(q, k, v, dO, bias, mask)
    loss(q, k, v, b, mask, dO) = sum(LTA._sdpa_default(q, k, v, mask, b) .* dO)
    dq, dk, dv = zero(q), zero(k), zero(v)
    db = bias === nothing ? nothing : zero(bias)
    Enzyme.autodiff(Reverse, loss, Active, Duplicated(q, dq), Duplicated(k, dk), Duplicated(v, dv),
        bias === nothing ? Const(nothing) : Duplicated(bias, db), Const(mask), Const(dO))
    return (; dq, dk, dv, dbias=db)
end

rows = String[]
println("device: ", DEVNAME, "   seeds: ", SEEDS, "   ratio = RMSE / (FP64 reference rounded once to the type)")
@printf("%-9s %-14s %-10s %-6s | %12s %12s %8s\n", "type", "shape", "setting", "grad", "rmse", "floor", "ratio")
for T in TYPES, (hd, H, n) in SHAPES, setting in ("plain", "bias+mask")
    hb = setting == "bias+mask"
    tune_flash_mma!(n; dim=hd, T)
    acc = Dict(g => (Float64[], Float64[]) for g in (:dq, :dk, :dv, :dbias))
    for seed in 1:SEEDS
        rng = Random.Xoshiro(1000 * seed + n)
        q, k, v, dO = operands(rng, hd, H, n)
        bias = hb ? randn(rng, Float64, n, n, H, 1, 1) : nothing
        mask = hb ? rand(rng, n, 1, 1, n, 1) .> 0.1 : nothing
        # round ONCE to the storage type, then everything is the upcast of those values
        rt(x) = x === nothing ? nothing : T.(x)
        qt, kt, vt, dOt, bt = rt(q), rt(k), rt(v), rt(dO), rt(bias)
        ref = reference(Float64.(qt), Float64.(kt), Float64.(vt), Float64.(dOt),
            bt === nothing ? nothing : Float64.(bt), mask)
        d(x) = x === nothing ? nothing : DEV(x)
        op, kern = LTA.SDPA(), LTA.FlashMMA()
        o, lse = KD.forward(op, kern, d(qt), d(kt), d(vt), d(mask), d(bt))
        g = KD.backward(op, kern, (d(qt), d(kt), d(vt), d(mask), d(bt)), (o, lse), (d(dOt), nothing))
        got = (; dq=g[1], dk=g[2], dv=g[3], dbias=g[5])
        for name in (:dq, :dk, :dv, :dbias)
            r = getproperty(ref, name)
            r === nothing && continue
            push!(acc[name][1], rmse(Array(getproperty(got, name)), r))
            push!(acc[name][2], rmse(T.(r), r))
        end
    end
    for name in (:dq, :dk, :dv, :dbias)
        isempty(acc[name][1]) && continue
        e, f = mean(acc[name][1]), mean(acc[name][2])
        @printf("%-9s %-14s %-10s %-6s | %12.4e %12.4e %7.2fx\n", nameof(T), "hd=$hd n=$n", setting, name, e, f, e / f)
        push!(rows, join((Dates.format(now(), "yyyy-mm-dd HH:MM"), DEVNAME, nameof(T), hd, H, n, setting, name, e, f, e / f), ","))
    end
end

new = !isfile(CSV)
open(CSV, "a") do io
    new && println(io, "when,device,type,hd,H,n,setting,grad,rmse,floor,ratio")
    foreach(r -> println(io, r), rows)
end
println("\nappended $(length(rows)) rows to $CSV")
