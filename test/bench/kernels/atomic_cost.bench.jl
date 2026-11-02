# How much of the backward is the pair-bias path (`dbias` atomics and the bias reads)? An UPPER BOUND on what reworking `dbias`
# can save, from existing binaries only.
#
#     KERNELDISPATCH_CACHE=$(mktemp -d) julia --project=test test/bench/kernels/atomic_cost.bench.jl
#     LTA_BENCH_GPU=cuda …      (default: Metal on a Mac, CUDA elsewhere)
#
# Phase 0 of the plan in docs/cuda_kernel_speed_improvements.md (improvement 10). Two arms on the same inputs, interleaved:
#   with bias   today: `:dkv` reads the bias and accumulates `dbias` with `_atomic_fadd!`, one per score;
#   no bias     no bias at all, a different binary (`HB=false`): no bias reads and no atomics.
# The ratio therefore bounds the atomic from ABOVE (it also drops the bias traffic). Isolating the atomic alone needs a kernel
# built without it, i.e. an edit of `flash_mma/backward.jl` (`_mma_kv_tile`, the `_atomic_fadd!` lines); that is a source
# change and belongs to the Phase 2 prototype, not here.

using LuxTriangleAttention, KernelDispatch, Random, Printf, Statistics

const LTA = LuxTriangleAttention
const KD = KernelDispatch

const GPU = lowercase(get(ENV, "LTA_BENCH_GPU", Sys.isapple() ? "metal" : "cuda"))
if GPU == "metal"
    using Metal
    const DEV = Metal.MtlArray
    sync() = Metal.synchronize()
    devname() = "Metal"
else
    using CUDA
    const DEV = CUDA.CuArray
    sync() = CUDA.synchronize()
    devname() = CUDA.name(CUDA.device())
end

const REPS = 10
const ROUNDS = 9
const SHAPES = [(32, 4, 256), (32, 4, 384), (16, 4, 384)]      # (hd, H, n)
const TYPES = (Float16, LTA.BFloat16)

# Wall clock around `REPS` synchronised calls: coarse next to GPU timestamps (Metal: see flash_mma.bench.jl), but the
# question here is a ratio of several milliseconds, and the same method serves both devices.
function timeit(f)
    f(); sync()
    t = time_ns()
    for _ in 1:REPS
        f()
    end
    sync()
    return (time_ns() - t) / REPS / 1e6
end

function problem(T, hd, H, n)
    rng = Random.Xoshiro(n)
    r(dims...) = DEV(T.(randn(rng, Float32, dims...)))
    return (; q=r(hd, H, n, n, 1), k=r(hd, H, n, n, 1), v=r(hd, H, n, n, 1), dO=r(hd, H, n, n, 1),
        bias=r(n, n, H, 1, 1), mask=DEV(rand(rng, n, 1, 1, n, 1) .> 0.1))
end

println("device: ", devname(), "   backward ms, median of $ROUNDS interleaved rounds of $REPS calls; ratio = with bias / no bias")
@printf("%-9s %-12s | %10s %10s | %10s\n", "type", "shape", "with bias", "no bias", "with/nobias")
op, kern = LTA.SDPA(), LTA.FlashMMA()
for T in TYPES, (hd, H, n) in SHAPES
    c = problem(T, hd, H, n)
    LTA.tune_flash_mma!(n; dim=hd, T)
    o, lse = KD.forward(op, kern, c.q, c.k, c.v, c.mask, c.bias)
    o0, lse0 = KD.forward(op, kern, c.q, c.k, c.v, c.mask, nothing)
    bwd_bias() = KD.backward(op, kern, (c.q, c.k, c.v, c.mask, c.bias), (o, lse), (c.dO, nothing))
    bwd_nobias() = KD.backward(op, kern, (c.q, c.k, c.v, c.mask, nothing), (o0, lse0), (c.dO, nothing))
    arms = Dict(:with => Float64[], :nobias => Float64[])
    for r in 1:ROUNDS
        for a in (isodd(r) ? (:with, :nobias) : (:nobias, :with))
            push!(arms[a], timeit(a === :with ? bwd_bias : bwd_nobias))
        end
    end
    m(a) = median(arms[a])
    @printf("%-9s %-12s | %10.3f %10.3f | %9.2fx\n", nameof(T), "hd=$hd n=$n", m(:with), m(:nobias), m(:with) / m(:nobias))
end
