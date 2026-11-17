# Does the output residual pay for 16-bit data? (TurboPairFormer, improvement 11 in docs/cuda_kernel_speed_improvements.md)
#
#     KERNELDISPATCH_CACHE=$(mktemp -d) julia --project=test test/bench/kernels/delta_residual.bench.jl
#     LTA_BENCH_GPU=cuda …      to pick the GPU (default: Metal on a Mac, CUDA elsewhere)
#
# The backward needs δ = Σ_d O·dO per query. The forward stores O rounded to the storage type, so δ computed from it carries that
# rounding. The paper's remedy: the forward also stores E = O32 − O16 (in the storage type) and the backward uses Σ dO·(O16 + E).
# This script measures the gain WITHOUT writing a kernel: it replaces `LTA._delta` and takes O32 from the Float64 reference forward
# (the kernel's own O32 carries a Float32 accumulation error besides, so this is a slight UPPER bound of the gain). Three δ per call:
#
#   base    Σ O16·dO                       (what the op does today)
#   resid   Σ (O16 + E)·dO, E = T(O64 − O16)   (the paper's scheme; shows what a 16-bit E loses, e.g. Float16 subnormals)
#   exact   Σ O64·dO                       (the ceiling: no rounding of O at all)
#
# Same protocol and error measure as `accuracy.bench.jl` (error / rounding floor of the Float64 reference); the three variants run
# on the SAME inputs, so the comparison is paired. `dv` does not use δ: it is the control and must not move.

using LuxTriangleAttention, KernelDispatch, Random, Printf, Statistics, Enzyme

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

const SHAPES = [(32, 4, 96), (32, 4, 192), (16, 4, 192)]
const TYPES = (Float16, LTA.BFloat16)
const SEEDS = 3

# The δ the op uses, switchable. Defined here, before the loop below runs, so the loop sees the new method.
const MODE = Ref(:base)
const EXTRA = Ref{Any}(nothing)        # E (resid) or O64 (exact), on the device
LTA._delta(o, dO) = begin
    x = MODE[] === :base ? LTA._f32(o) : MODE[] === :resid ? LTA._f32(o) .+ LTA._f32(EXTRA[]) : LTA._f32(EXTRA[])
    permutedims(sum(x .* LTA._f32(dO); dims=1), (1, 3, 2, 4, 5))
end

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

const MODES = (:base, :resid, :exact)
GRADS = (:dq, :dk, :dv, :dbias)
println("device: ", DEVNAME, "   seeds: ", SEEDS, "   ratio = RMSE / (FP64 reference rounded once to the type)")
@printf("%-9s %-12s %-10s %-6s | %8s %8s %8s | %s\n", "type", "shape", "setting", "grad", "base", "resid", "exact", "resid/base  exact/base")
for T in TYPES, (hd, H, n) in SHAPES, setting in ("plain", "bias+mask")
    hb = setting == "bias+mask"
    tune_flash_mma!(n; dim=hd, T)
    acc = Dict((m, g) => Float64[] for m in MODES, g in GRADS)
    flo = Dict(g => Float64[] for g in GRADS)
    for seed in 1:SEEDS
        rng = Random.Xoshiro(1000 * seed + n)
        q, k, v, dO = operands(rng, hd, H, n)
        bias = hb ? randn(rng, Float64, n, n, H, 1, 1) : nothing
        mask = hb ? rand(rng, n, 1, 1, n, 1) .> 0.1 : nothing
        rt(x) = x === nothing ? nothing : T.(x)
        qt, kt, vt, dOt, bt = rt(q), rt(k), rt(v), rt(dO), rt(bias)
        f64(x) = x === nothing ? nothing : Float64.(x)
        ref = reference(f64(qt), f64(kt), f64(vt), f64(dOt), f64(bt), mask)
        o64 = LTA._sdpa_default(f64(qt), f64(kt), f64(vt), mask, f64(bt))
        d(x) = x === nothing ? nothing : DEV(x)
        op, kern = LTA.SDPA(), LTA.FlashMMA()
        o, lse = KD.forward(op, kern, d(qt), d(kt), d(vt), d(mask), d(bt))
        E = T.(o64 .- Float64.(Array(o)))
        for g in GRADS
            r = getproperty(ref, g); r === nothing || push!(flo[g], rmse(T.(r), r))
        end
        for m in MODES
            MODE[] = m
            EXTRA[] = m === :resid ? d(E) : m === :exact ? d(Float32.(o64)) : nothing
            gr = KD.backward(op, kern, (d(qt), d(kt), d(vt), d(mask), d(bt)), (o, lse), (d(dOt), nothing))
            got = (; dq=gr[1], dk=gr[2], dv=gr[3], dbias=gr[5])
            for g in GRADS
                r = getproperty(ref, g); r === nothing && continue
                push!(acc[(m, g)], rmse(Array(getproperty(got, g)), r))
            end
        end
        MODE[] = :base
    end
    for g in GRADS
        isempty(flo[g]) && continue
        f = mean(flo[g]); b, rs, ex = (mean(acc[(m, g)]) / f for m in MODES)
        @printf("%-9s %-12s %-10s %-6s | %7.2fx %7.2fx %7.2fx | %6.2f  %6.2f\n", nameof(T), "hd=$hd n=$n", setting, g, b, rs, ex, rs / b, ex / b)
    end
end
