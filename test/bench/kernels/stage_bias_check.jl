# The forward with the bias tile staged in local memory (`_mma_stage_bias`), checked on a device whose own default is "not
# staged" (Metal): the staging code is backend-agnostic, so this tests it without a CUDA GPU. NOT a timing: on Metal the
# staged path is not tuned or selected, only forced here.
#
#     KERNELDISPATCH_CACHE=$(mktemp -d) julia --project=test test/bench/kernels/stage_bias_check.jl
#
# A fresh cache matters: the hook is not part of a binary's name, so a cache from a run without the override would be reused.

using LuxTriangleAttention, KernelDispatch, Random, Metal, Printf

const LTA = LuxTriangleAttention
const KD = KernelDispatch

LTA._mma_stage_bias(::KD.MetalBackendTag) = true
@assert LTA._mma_stage_bias(KD.device_backend())
# `LTA_CHECK_BREAK=1`: make the staged read return zeros, to show the check does exercise it (the errors must then jump)
get(ENV, "LTA_CHECK_BREAK", "0") == "1" && @eval LTA @inline _mma_bias2s(bs, rl, c, ::Val{HB}) where {HB} = (0f0, 0f0)

relL2(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))

op, kern = LTA.SDPA(), LTA.FlashMMA()
worst = 0.0
for T in (Float16, Float32), (hd, H, n, nouter, B) in ((32, 2, 37, 5, 2), (16, 4, 24, 24, 1), (32, 2, 70, 3, 1)),
    hm in (true, false)
    rng = Random.Xoshiro(n + hd)
    r(dims...) = T.(randn(rng, Float32, dims...))
    q, k, v = (r(hd, H, n, nouter, B) for _ in 1:3)
    bias = r(n, n, H, 1, B)
    mask = hm ? rand(rng, n, 1, 1, nouter, B) .> 0.2 : nothing
    LTA.tune_flash_mma!(n; dim=hd, T)
    d(x) = x === nothing ? nothing : Metal.MtlArray(x)
    o, _ = KD.forward(op, kern, d(q), d(k), d(v), d(mask), d(bias))
    ref = LTA._sdpa_default(Float64.(q), Float64.(k), Float64.(v), mask, Float64.(bias))
    e = relL2(Array(o), ref)
    global worst = max(worst, e)
    @printf("%-8s hd=%-3d n=%-3d nouter=%-3d mask=%-5s relL2 %.2e\n", nameof(T), hd, n, nouter, hm, e)
end
println(worst < 3e-3 ? "OK: staged bias matches the reference (worst $(worst))" : "FAIL: worst relL2 $worst")
