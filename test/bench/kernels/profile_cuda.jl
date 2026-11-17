# One `:flash_mma` forward and one backward, for Nsight Compute. Tuning and warm-up run first and are NOT profiled: the
# profiler only sees the region between `CUDA.Profile.start()` and `stop()` (run ncu with `--profile-from-start off`).
#
#   export KERNELDISPATCH_CACHE=$HOME/lta_prof_cache          # a private cache; keep it between the two runs
#   julia --project=test test/bench/kernels/profile_cuda.jl   # 1st run: tunes and caches (no profiler)
#   ncu --set full --profile-from-start off -k regex:lta_flash_mma -o lta_prof \
#       julia --project=test test/bench/kernels/profile_cuda.jl
#
# Shape and type: `LTA_PROF=hd,n,type` (default 32,384,Float16). Kernels: lta_flash_mma_fwd, _bwd_kv, _bwd_q.

using CUDA, KernelDispatch, LuxTriangleAttention, Random

const LTA = LuxTriangleAttention
const KD = KernelDispatch

spec = split(get(ENV, "LTA_PROF", "32,384,Float16"), ",")
hd, n = parse(Int, spec[1]), parse(Int, spec[2])
T = spec[3] == "BFloat16" ? LTA.BFloat16 : getfield(Base, Symbol(spec[3]))
H, B = 4, 1

rng = Random.Xoshiro(1)
r(dims...) = CuArray(T.(randn(rng, Float32, dims...)))
q, k, v, dO = (r(hd, H, n, n, B) for _ in 1:4)
bias = r(n, n, H, 1, B)
mask = CuArray(rand(rng, n, 1, 1, n, B) .> 0.1)

LTA.tune_flash_mma!(n; dim=hd, T)
op, kern = LTA.SDPA(), LTA.FlashMMA()
fwd() = KD.forward(op, kern, q, k, v, mask, bias)
res = fwd()                                       # (o, lse), plus `e` with the `output_residual` preference
bwd() = KD.backward(op, kern, (q, k, v, mask, bias), res, (dO, nothing))
bwd(); CUDA.synchronize()                         # warm-up: compiles and loads everything

CUDA.Profile.start()
fwd(); bwd()
CUDA.synchronize()
CUDA.Profile.stop()
println("profiled one forward and one backward: hd=$hd n=$n $T on ", CUDA.name(CUDA.device()))
