# The tensor-core fragment layout `:flash_mma` depends on (ext/LuxTriangleAttentionCUDAExt/flash_mma.jl).
#
# NVIDIA documents the `mma.sync` m16n8k16 register layout, but the kernel's in-register softmax mixes rows silently if
# a hook disagrees with it, so it is pinned here through the very hooks the kernel calls: one warp loads `A` (16×16)
# and two `B` blocks (16×8) from local memory, forms the two scores `S1`, `S2` (16×8 each), reduces every row of
# `[S1 S2]` across the lanes that hold it, rounds the scores into the next product's operand (`_mma_pfrag`) and
# multiplies it by a `V` block. All of it is compared with the same arithmetic on the host.

using LuxTriangleAttention.KernelAbstractions
import KernelDispatch
import LuxTriangleAttention as LTA
using KernelDispatch: OutArray, run_binary

# One row slot `s` of the two scores: each lane stores its two entries of S (a 16×16 column-major matrix `[S1 S2]`)
# and the row's maximum and sum over all 16 columns. `s` is a literal at the call sites, so the slot index folds.
@inline function _probe_slot!(be, cout, redout, C1, C2, fr, col, ::Val{s}) where {s}
    row = fr + Int32(8 * (s - 1))
    x1, y1 = LTA._mma_pair(be, C1, s)
    x2, y2 = LTA._mma_pair(be, C2, s)
    unsafe_store!(cout, x1, row + Int32(16) * col + Int32(1))
    unsafe_store!(cout, y1, row + Int32(16) * (col + Int32(1)) + Int32(1))
    unsafe_store!(cout, x2, row + Int32(16) * (col + Int32(8)) + Int32(1))
    unsafe_store!(cout, y2, row + Int32(16) * (col + Int32(9)) + Int32(1))
    unsafe_store!(redout, LTA._mma_rowmax(be, max(max(x1, y1), max(x2, y2))), 2row + Int32(1))
    unsafe_store!(redout, LTA._mma_rowsum(be, (x1 + y1) + (x2 + y2)), 2row + Int32(2))
    return nothing
end

@inline function _probe_out!(be, oout, O, fr, col, ::Val{s}) where {s}
    row = fr + Int32(8 * (s - 1))
    x, y = LTA._mma_pair(be, O, s)
    unsafe_store!(oout, x, row + Int32(16) * col + Int32(1))
    unsafe_store!(oout, y, row + Int32(16) * (col + Int32(1)) + Int32(1))
    return nothing
end

@kernel unsafe_indices = true function _cuda_mma_probe!(cout, redout, oout, a, b, v, ::Val{E}) where {E}
    be = KernelDispatch.CUDABackendTag()
    l = LTA._i32(@index(Local, Linear)) - Int32(1)
    aS = @localmem E (16, 16)
    bS = @localmem E (16, 16)
    vS = @localmem E (16, 8)
    for k in Int32(0):Int32(7)
        e = l + Int32(32) * k
        @inbounds aS[e%Int32(16)+Int32(1), e÷Int32(16)+Int32(1)] = unsafe_load(a, e + Int32(1))
        @inbounds bS[e%Int32(16)+Int32(1), e÷Int32(16)+Int32(1)] = unsafe_load(b, e + Int32(1))
    end
    for k in Int32(0):Int32(3)
        e = l + Int32(32) * k
        @inbounds vS[e%Int32(16)+Int32(1), e÷Int32(16)+Int32(1)] = unsafe_load(v, e + Int32(1))
    end
    @synchronize
    fr, col = LTA._mma_lane(be, l)
    zero4 = LTA._mma_zero(be, Float32)
    A = LTA._mma_load_a(be, aS, 1, 1)
    C1 = LTA._mma(be, A, LTA._mma_load_b(be, bS, 1, 1), zero4)
    C2 = LTA._mma(be, A, LTA._mma_load_b(be, bS, 1, 9), zero4)
    _probe_slot!(be, cout, redout, C1, C2, fr, col, Val(1))
    _probe_slot!(be, cout, redout, C1, C2, fr, col, Val(2))
    # The scores as the next operand: blocks j = 0, 1 of row slots 1, 2, in the order `_mma_fwd_tile` lists them.
    P = LTA._mma_pfrag(be, E, LTA._mma_pair(be, C1, 1), LTA._mma_pair(be, C1, 2),
        LTA._mma_pair(be, C2, 1), LTA._mma_pair(be, C2, 2))
    O = LTA._mma(be, P, LTA._mma_load_b(be, vS, 1, 1), zero4)
    _probe_out!(be, oout, O, fr, col, Val(1))
    _probe_out!(be, oout, O, fr, col, Val(2))
end

@testset "flash_mma tensor-core fragment layout" for E in (Float16, LTA.BFloat16)
    rng = Random.Xoshiro(3)
    r(dims...) = E.(randn(rng, Float32, dims...) ./ 4)
    A, B, V = r(16, 16), r(16, 16), r(16, 8)                      # B = [B1 B2]
    f32(x) = Float32.(x)
    bin = KernelDispatch.ka_compile(KernelDispatch.CUDABackendTag(), _cuda_mma_probe!,
        (AbstractArray{Float32}, AbstractArray{Float32}, AbstractArray{Float32},
            AbstractArray{E}, AbstractArray{E}, AbstractArray{E}, Val{E});
        tg=32, name="cuda_mma_probe_$(nameof(E))_" * string(hash(read(@__FILE__)); base=36), dir=mktempdir())
    cout, red, oout = run_binary(bin, OutArray(Float32, 256), OutArray(Float32, 32), OutArray(Float32, 128),
        DeviceArray(vec(A)), DeviceArray(vec(B)), DeviceArray(vec(V)); extent=32)      # one warp: one group
    S = reshape(Array(cout), 16, 16)
    RED = reshape(Array(red), 2, 16)
    O = reshape(Array(oout), 16, 8)

    S_ref = f32(A) * f32(B)                                       # 16×16: [S1 S2]
    @test S ≈ S_ref rtol = 1e-5 atol = 1e-6                       # every lane's entries sit in the right place
    @test RED[1, :] ≈ vec(maximum(S_ref; dims=2)) rtol = 1e-5     # the row reductions see all 16 columns
    @test RED[2, :] ≈ vec(sum(S_ref; dims=2)) rtol = 1e-4 atol = 1e-5
    P_ref = f32(E.(S_ref))                                        # rounded to the operand type, as `_mma_pfrag` does
    @test O ≈ P_ref * f32(V) rtol = 1e-4 atol = 1e-5              # a 16-wide contraction over the two score blocks
end
