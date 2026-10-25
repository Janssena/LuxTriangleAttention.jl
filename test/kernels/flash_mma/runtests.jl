# Mirrors `src/kernels/flash_mma/`. Metal is the vehicle, as for `flash/`: what is under test is the
# kernel's math. The fragment lane layout it relies on is tested on its own, with the backend, in
# `test/ext/metal/flash_mma.test.jl`.
#
# Shared setup for `forward.gpu.test.jl` and `backward.gpu.test.jl`: the case grid and its operands, how the
# kernel is run, and the ground truth.
#
# The kernel is run by calling its KernelDispatch methods DIRECTLY (`mma_forward`/`mma_backward`), after an
# explicit `tune_flash_mma!`, never through `use_kernel!` + `SDPA()`: a selection that silently
# did nothing would route to `:default` and every test would pass against the wrong kernel. The
# routing itself is checked once, separately, in `forward.gpu.test.jl`.

using Test, Random, Enzyme
import KernelDispatch

relL2_mma(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))
# relL2 bounds per storage type: the output's own rounding (8-bit mantissa for BFloat16), plus the
# 16-bit operands the tuner may choose (`ops`).
tol_fwd(T) = T === Float32 ? 1e-5 : T === Float16 ? 2e-3 : 1e-2
tol_bwd(T) = T === Float32 ? 1e-5 : T === Float16 ? 1e-2 : 3e-2
nanless(x) = map(y -> isnan(y) ? 0.0 : Float64(y), x)

# The storage types the loaded device's matrix units take as they are. Others run at another type (CUDA: Float32 at Float16,
# `CAST_TYPES`: rounded inputs, results cast back), tested on their own in `forward.gpu.test.jl` / `backward.gpu.test.jl`.
const MMA_ALL_TYPES = (Float32, Float16, LuxTriangleAttention.BFloat16)
const MMA_TYPES = filter(T -> LuxTriangleAttention.mma_compute_type(LuxTriangleAttention.mma_backend(), T) === T, MMA_ALL_TYPES)
const CAST_TYPES = filter(MMA_ALL_TYPES) do T
    Tc = LuxTriangleAttention.mma_compute_type(LuxTriangleAttention.mma_backend(), T)
    Tc !== nothing && Tc !== T
end

const MMA_CASES = [(T, shape, hb, hm)
    for T in MMA_TYPES, shape in ((32, 2, 37, 5, 2), (16, 4, 24, 24, 1)),
        hb in (true, false), hm in (true, false)]

# Operands of one case: `[hd, H, n, nouter, B]` q/k/v/dO, an optional bias and a KEEP mask with one
# fully masked (i, b) slice; `d` moves an operand to the device.
function mma_case(T, shape, hb, hm)
    hd, H, n, nouter, B = shape
    rng = Random.Xoshiro(n + hd)
    r(dims...) = T.(randn(rng, Float32, dims...))      # `randn` has no BFloat16 method
    q, k, v, dO = (r(hd, H, n, nouter, B) for _ in 1:4)
    bias = hb ? r(n, n, H, 1, B) : nothing
    mask = hm ? rand(rng, n, 1, 1, nouter, B) .> 0.2 : nothing
    hm && (mask[:, 1, 1, 2, 1] .= false)
    return (; q, k, v, dO, bias, mask, d=x -> x === nothing ? nothing : DeviceArray(x))
end

const MMA = () -> (LTA_.SDPA(), LTA_.FlashMMA())

# `:flash_mma`'s forward and backward on one case, at the case's own storage type.
function mma_forward(c, T, shape)
    hd, _, n = shape
    tune_flash_mma!(n; dim=hd, T)
    op, kern = MMA()
    Tc = LTA_.mma_compute_type(LTA_.mma_backend(), T)       # the type the call runs at (T, unless the device casts)
    @test haskey(KernelDispatch.variants(op, kern), LTA_._attention_variant_key(kern, Tc, hd, n;
        hasbias=c.bias !== nothing, hasmask=c.mask !== nothing))
    return KernelDispatch.forward(op, kern, c.d(c.q), c.d(c.k), c.d(c.v), c.d(c.mask), c.d(c.bias))
end

function mma_backward(c, o, lse)
    op, kern = MMA()
    g = KernelDispatch.backward(op, kern, (c.d(c.q), c.d(c.k), c.d(c.v), c.d(c.mask), c.d(c.bias)),
        (o, lse), (c.d(c.dO), nothing))
    return g[1], g[2], g[3], g[5]                       # dq, dk, dv, dbias
end

# Ground truth for the backward: Enzyme's reverse-mode gradient of `_sdpa_default` for the loss
# `Σ o·dO`, on the host at Float64 and on the same (upcast) values. A fully masked (i, b) slice attends
# uniformly (`_with_dead_rows`): its `dV` is `dO / n` per key and everything else is zero, which is what
# the kernel is specified to write.
function mma_reference_grads(c)
    q, k, v, dO = Float64.(c.q), Float64.(c.k), Float64.(c.v), Float64.(c.dO)
    bias = c.bias === nothing ? nothing : Float64.(c.bias)
    mask = c.mask === nothing ? nothing : copy(c.mask)
    loss(q, k, v, b, mask, dO) = sum(LuxTriangleAttention._sdpa_default(q, k, v, mask, b) .* dO)
    dq, dk, dv = zero(q), zero(k), zero(v)
    db = bias === nothing ? nothing : zero(bias)
    Enzyme.autodiff(Reverse, loss, Active, Duplicated(q, dq), Duplicated(k, dk), Duplicated(v, dv),
        bias === nothing ? Const(nothing) : Duplicated(bias, db), Const(mask), Const(dO))
    return dq, dk, dv, db
end

if GPU !== nothing
    LTA_ = LuxTriangleAttention
    prev = LTA_.current_kernel()
    try
        @testset verbose = VERBOSE "forward" begin
            include("forward.gpu.test.jl")
        end

        @testset verbose = VERBOSE "backward" begin
            include("backward.gpu.test.jl")
        end

        @testset verbose = VERBOSE "tune" begin
            include("tune.gpu.test.jl")
        end
    finally
        use_kernel!(prev)
    end
end
