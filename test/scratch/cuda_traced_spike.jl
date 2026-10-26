# Why do the traced (Reactant) tests fail on CUDA? Two suspects, told apart here:
#   1. XLA's own Float32 matrix products on Ampere (TF32 at the default precision): affects plain `_sdpa_default`
#      compiled by Reactant, no kernel involved. Compared at the default and at HIGHEST precision.
#   2. The traced kernel custom call returning wrong values. Compared against the host reference AND against the SAME
#      kernel launched eagerly on CuArrays, slice by slice, so a launch-geometry problem shows as a pattern.
#
#   KERNELDISPATCH_CACHE=$(mktemp -d) julia --project=test test/scratch/cuda_traced_spike.jl 2>&1 | tee lta_cuda_traced_spike.log

using CUDA, Reactant, KernelDispatch, LuxTriangleAttention, Random, Printf

const LTA = LuxTriangleAttention
Reactant.set_default_backend("gpu")

relerr(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))

T = Float32
hd, H, N, B = 8, 4, 8, 1
rng = Random.Xoshiro(42)
q, k, v = (randn(rng, T, hd, H, N, N, B) for _ in 1:3)
bias = randn(rng, T, N, N, H, 1, B)
mask = fill(true, N, 1, 1, N, B)
mask[2, :, :, 3, :] .= false

LTA.tune_flash_kernel!(N; dim=hd, T)               # a trace cannot tune

ref = LTA._sdpa_default(q, k, v, mask, bias)       # host, Float32
dev = Reactant.to_rarray
qd, kd, vd, md, bd = dev(q), dev(k), dev(v), dev(mask), dev(bias)

println("--- 1. plain XLA ops (no kernel): compiled _sdpa_default against the host")
fplain(q, k, v, m, b) = LTA._sdpa_default(q, k, v, m, b)
plain = Array(Reactant.@jit fplain(qd, kd, vd, md, bd))
plain_hi = Reactant.with_config(; dot_general_precision=Reactant.PrecisionConfig.HIGHEST) do
    Array(Reactant.@jit fplain(qd, kd, vd, md, bd))
end
@printf("    default precision: relative error %.3e\n    HIGHEST precision: relative error %.3e\n",
    relerr(plain, ref), relerr(plain_hi, ref))

println("--- 2. the kernel: eager on CuArrays, and traced as a custom call")
fo(q, k, v, m, b) = first(LTA.SDPA()(q, k, v, m, b))
eager = Array(first(LTA.SDPA()(CuArray(q), CuArray(k), CuArray(v), CuArray(mask), CuArray(bias))))
traced = Array(Reactant.@jit fo(qd, kd, vd, md, bd))
@printf("    eager  vs host: %.3e\n    traced vs host: %.3e\n    traced vs eager: %.3e\n",
    relerr(eager, ref), relerr(traced, ref), relerr(traced, eager))
println("    traced vs eager, per (outer index j = dim 4, batch = dim 5) slice [relative error]:")
for j in 1:N, b in 1:B
    e = relerr(traced[:, :, :, j, b], eager[:, :, :, j, b])
    @printf("      j=%d b=%d  %.3e %s\n", j, b, e, e > 1e-4 ? "  <-- differs" : "")
end
println("    traced vs eager, per head (dim 2):")
for h in 1:H
    @printf("      head %d  %.3e\n", h, relerr(traced[:, h, :, :, :], eager[:, h, :, :, :]))
end

println("--- 3. the custom call in the IR")
ir = repr(Reactant.@code_hlo optimize = false fo(qd, kd, vd, md, bd))
for l in split(ir, '\n')
    occursin("custom_call", l) && println("    ", first(l, 1500))
end
