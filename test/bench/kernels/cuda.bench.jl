# `:default` against `:flash_mma` on an NVIDIA GPU, forward and backward, on triangle-attention shapes.
#
#   KERNELDISPATCH_CACHE=$(mktemp -d) julia --project=test test/bench/kernels/cuda.bench.jl 2>&1 | tee lta_cuda_bench.log
#   LTA_BENCH=hd8 …   only the head-dimension-8 table (the quick one): is a zero-padded `:flash_mma` faster than `:default`?
#
# Each cell is the GPU time of one call (CUDA events, mean of `REPS` after a warm-up), through the kernels' KernelDispatch
# methods directly. The kernels are tuned first (`tune_flash_kernel!`, `tune_flash_mma!`), outside the timed region.
# Float32 `:flash_mma` runs at Float16 on CUDA (`mma_compute_type`); a head dimension that is not a multiple of 16 runs
# `:default` there, and the cell says so.

using CUDA, KernelDispatch, LuxTriangleAttention, Random, Printf

const LTA = LuxTriangleAttention
const KD = KernelDispatch

const REPS = 10
const H, B = 4, 1
const LENGTHS = (128, 256, 384)
const HEADDIMS = (16, 32, 64)
const TYPES = (Float32, Float16, LTA.BFloat16)

gputime(f) = (f(); f(); CUDA.synchronize(); CUDA.@elapsed(begin
    for _ in 1:REPS
        f()
    end
end) / REPS * 1e3)                                  # ms

function problem(T, hd, n)
    rng = Random.Xoshiro(1)
    r(dims...) = CuArray(T.(randn(rng, Float32, dims...)))
    q, k, v, dO = (r(hd, H, n, n, B) for _ in 1:4)
    bias = r(n, n, H, 1, B)
    mask = CuArray(rand(rng, n, 1, 1, n, B) .> 0.1)
    return (; q, k, v, dO, bias, mask)
end

op = LTA.SDPA()
kernels = (default=LTA.FlashScalar(), mma=LTA.FlashMMA())

println("device: ", CUDA.name(CUDA.device()), "   H=$H, B=$B, triangle shapes [hd, H, n, n, B], bias and mask present")
@printf("%-9s %4s %4s | %10s %10s %7s | %10s %10s %7s | %s\n", "type", "hd", "n", "fwd dflt", "fwd mma", "x", "bwd dflt", "bwd mma", "x", "mma config (fwd)")
for T in TYPES, hd in HEADDIMS, n in LENGTHS
    get(ENV, "LTA_BENCH", "all") == "hd8" && break         # LTA_BENCH=hd8: only the head-dimension-8 table below
    c = problem(T, hd, n)
    LTA.tune_flash_kernel!(n; dim=hd, T)
    LTA.tune_flash_mma!(n; dim=hd, T)
    row = Dict{Symbol,Tuple{Float64,Float64}}()
    for (name, kern) in pairs(kernels)
        fwd() = KD.forward(op, kern, c.q, c.k, c.v, c.mask, c.bias)
        o, lse = fwd()
        bwd() = KD.backward(op, kern, (c.q, c.k, c.v, c.mask, c.bias), (o, lse), (c.dO, nothing))
        row[name] = (gputime(fwd), gputime(bwd))
    end
    cfg = try
        LTA.flash_mma_config(LTA.mma_compute_type(LTA.mma_backend(), T), hd, n)[:fwd]
    catch
        "runs :default"
    end
    @printf("%-9s %4d %4d | %9.3f  %9.3f  %6.2fx | %9.3f  %9.3f  %6.2fx | %s\n", nameof(T), hd, n,
        row.default[1], row.mma[1], row.default[1] / row.mma[1],
        row.default[2], row.mma[2], row.default[2] / row.mma[2], cfg)
end

# --- head dimension 8 (the AlphaFold size): would padding it to 16 pay off? -------------------------------------------------
#
# The matrix units contract over 16 channels, so `:flash_mma` does not run head dimension 8 yet (it falls back to `:default`).
# The cost of padding is measured here by zero-padding the inputs to 16 channels and running the EXISTING `:flash_mma` at
# head dimension 16: the extra channels are zeros, so it is the same attention. That is a CONSERVATIVE estimate of a real
# padded kernel, which would still load 8 channels and run the `P·V` product over 8 output channels, where this runs both
# over 16. "dflt@8" is `:default` at the real head dimension 8, "dflt@16" the same kernel at the padded size, for scale.
println("\nhead dimension 8: `:default` at 8 against `:flash_mma` on inputs zero-padded to 16 (an upper bound on a padded kernel's time)")
@printf("%-9s %4s | %10s %10s %10s %7s | %10s %10s %10s %7s\n", "type", "n", "fwd dflt@8", "fwd dflt@16", "fwd mma@16", "x",
    "bwd dflt@8", "bwd dflt@16", "bwd mma@16", "x")
pad(x) = cat(x, CUDA.zeros(eltype(x), 8, size(x)[2:end]...); dims=1)       # [8, …] -> [16, …]
for T in TYPES, n in LENGTHS
    c8 = problem(T, 8, n)
    c16 = merge(c8, (; q=pad(c8.q), k=pad(c8.k), v=pad(c8.v), dO=pad(c8.dO)))
    LTA.tune_flash_kernel!(n; dim=8, T)
    LTA.tune_flash_kernel!(n; dim=16, T)
    LTA.tune_flash_mma!(n; dim=16, T)
    t = Dict{Symbol,Tuple{Float64,Float64}}()
    for (label, c, kern) in ((:d8, c8, kernels.default), (:d16, c16, kernels.default), (:m16, c16, kernels.mma))
        fwd() = KD.forward(op, kern, c.q, c.k, c.v, c.mask, c.bias)
        o, lse = fwd()
        bwd() = KD.backward(op, kern, (c.q, c.k, c.v, c.mask, c.bias), (o, lse), (c.dO, nothing))
        t[label] = (gputime(fwd), gputime(bwd))
    end
    @printf("%-9s %4d | %9.3f  %9.3f  %9.3f  %6.2fx | %9.3f  %9.3f  %9.3f  %6.2fx\n", nameof(T), n,
        t[:d8][1], t[:d16][1], t[:m16][1], t[:d8][1] / t[:m16][1],
        t[:d8][2], t[:d16][2], t[:m16][2], t[:d8][2] / t[:m16][2])
end
