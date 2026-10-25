# CUDA's half of `:flash_mma` (src/kernels/flash_mma/): the fragment hooks on tensor cores, through `mma.sync`
# m16n8k16 (Ampere, sm_80, and newer), dispatched on KernelDispatch' `CUDABackendTag`. Building, verifying, timing and
# tuning the stages is backend-agnostic (`src/kernels/launch.jl`, `src/kernels/tune.jl`).
#
# One product is `C[16×8] += A[16×16]·B[16×8]`; with g = lane÷4 and t = lane%4, a lane holds (the layout is
# pinned by `test/ext/cuda/flash_mma.test.jl`, and was measured by `test/scratch/cuda_spike.jl`):
#
#   C (4 × Float32)   c0, c1 = C[g, 2t], C[g, 2t+1]          c2, c3 = C[g+8, 2t], C[g+8, 2t+1]
#   A (4 × 2 packed)  a0 = A[g, 2t..2t+1]    a1 = A[g+8, 2t..]    a2 = A[g, 2t+8..2t+9]    a3 = A[g+8, 2t+8..]
#   B (2 × 2 packed)  b0 = B[2t..2t+1, g]    b1 = B[2t+8..2t+9, g]
#
# so an accumulator is TWO row slots (rows g and g+8) of two adjacent columns, and two adjacent n8 accumulators,
# converted to 16 bits, are exactly one `A` operand (`_mma_pfrag`): the probabilities go from the softmax to the
# next product without leaving registers. The 16-bit operands are Float16 or BFloat16, products always accumulate in
# Float32. There is no Float32 operand (that would be TF32, whose 10-bit mantissa is not accurate enough): Float32
# storage runs `:default`.

using KernelDispatch: CUDABackendTag

const Interop = CUDACore.LLVM.Interop
const BF16 = LTA.BFloat16
const F16OrBF16 = Union{Float16,BF16}

# Operands are two packed 16-bit values per register; the element type rides along to pick the instruction.
struct AFrag{E}
    r::NTuple{4,UInt32}
end
struct BFrag{E}
    r::NTuple{2,UInt32}
end

# The 16×8×16 product of Ampere's tensor cores.
LTA._mma_shape(::CUDABackendTag) = (16, 8, 16)

# Only 16-bit operands, and only where `mma.sync` m16n8k16 exists (sm_80+): the tuner offers these for `ops`.
LTA.mma_operand_types(::CUDABackendTag, ::Type{T}) where {T} =
    T <: F16OrBF16 && CUDACore.capability(CUDACore.device()) >= v"8.0" ? DataType[T] : DataType[]

# The tensor cores take Float16 and BFloat16 storage as it is. Float32 storage runs at Float16 (its inputs saturating-cast,
# its results cast back to Float32), for the accuracy of TF32 (also an 11-bit significand) at half the memory traffic of
# Float32 storage; `:default` keeps Float32 exact. Anything else, or a GPU older than sm_80, has no matrix-unit path.
LTA.mma_compute_type(::CUDABackendTag, ::Type{T}) where {T} =
    CUDACore.capability(CUDACore.device()) < v"8.0" ? nothing :
    T <: F16OrBF16 ? T : T === Float32 ? Float16 : nothing

# The lane within its warp, 0-based (the groups are 1-D: `threads = (32·NSG, 1, 1)`).
@inline _lane() = (CUDACore.threadIdx().x - Int32(1)) & Int32(31)

@inline _bits(x::F16OrBF16) = UInt32(reinterpret(UInt16, x))
@inline _pack(x::E, y::E) where {E<:F16OrBF16} = _bits(x) | (_bits(y) << UInt32(16))

# Matrix `a` is column-major; `A[row, col]` is `a[r + row, c + col]`, one 32-bit register per two entries of a row.
@inline function LTA._mma_load_a(::CUDABackendTag, a, r, c)
    g = _lane() >> Int32(2)
    t = _lane() & Int32(3)
    E = eltype(a)
    @inbounds begin
        a0 = _pack(a[r+g, c+2t], a[r+g, c+2t+1])
        a1 = _pack(a[r+g+8, c+2t], a[r+g+8, c+2t+1])
        a2 = _pack(a[r+g, c+2t+8], a[r+g, c+2t+9])
        a3 = _pack(a[r+g+8, c+2t+8], a[r+g+8, c+2t+9])
    end
    return AFrag{E}((a0, a1, a2, a3))
end

# `B[k, n]` is `a[r + k, c + n]`: the two entries of a register are adjacent in memory.
@inline function LTA._mma_load_b(::CUDABackendTag, a, r, c)
    g = _lane() >> Int32(2)
    t = _lane() & Int32(3)
    E = eltype(a)
    @inbounds begin
        b0 = _pack(a[r+2t, c+g], a[r+2t+1, c+g])
        b1 = _pack(a[r+2t+8, c+g], a[r+2t+9, c+g])
    end
    return BFrag{E}((b0, b1))
end

@inline LTA._mma_zero(::CUDABackendTag, ::Type{Float32}) = (0.0f0, 0.0f0, 0.0f0, 0.0f0)

for (E, ptx) in ((Float16, "f16"), (BF16, "bf16"))
    asm = "mma.sync.aligned.m16n8k16.row.col.f32.$ptx.$ptx.f32 {\$0,\$1,\$2,\$3}, {\$4,\$5,\$6,\$7}, {\$8,\$9}, " *
          "{\$10,\$11,\$12,\$13};"
    @eval @inline function LTA._mma(::CUDABackendTag, a::AFrag{$E}, b::BFrag{$E}, c::NTuple{4,Float32})
        return Interop.@asmcall($asm, "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", true, NTuple{4,Float32},
            Tuple{UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,Float32,Float32,Float32,Float32},
            a.r[1], a.r[2], a.r[3], a.r[4], b.r[1], b.r[2], c[1], c[2], c[3], c[4])
    end
end

# Two Float32 into one register of two 16-bit values (`x` low): one conversion instruction, round to nearest.
@inline _pack2(::Type{Float16}, x::Float32, y::Float32) = Interop.@asmcall("cvt.rn.f16x2.f32 \$0, \$2, \$1;",
    "=r,f,f", false, UInt32, Tuple{Float32,Float32}, x, y)
@inline _pack2(::Type{BF16}, x::Float32, y::Float32) = Interop.@asmcall("cvt.rn.bf16x2.f32 \$0, \$2, \$1;",
    "=r,f,f", false, UInt32, Tuple{Float32,Float32}, x, y)

# The next product's `A` operand from four accumulator pairs: blocks of 8 columns j = 0, 1 (the two halves of the
# 16-wide contraction), row slots 1, 2 (rows g, g+8) — `a0 = (j0, slot 1)`, `a1 = (j0, slot 2)`, `a2 = (j1, slot 1)`,
# `a3 = (j1, slot 2)`.
@inline LTA._mma_pfrag(::CUDABackendTag, ::Type{E}, p1, p2, p3, p4) where {E<:F16OrBF16} = AFrag{E}((
    _pack2(E, Float32(p1[1]), Float32(p1[2])), _pack2(E, Float32(p2[1]), Float32(p2[2])),
    _pack2(E, Float32(p3[1]), Float32(p3[2])), _pack2(E, Float32(p4[1]), Float32(p4[2]))))

# Row slot `s` (1, 2) of an accumulator is its elements `2s-1`, `2s`.
@inline LTA._mma_pair(::CUDABackendTag, f::NTuple{4,Float32}, s::Integer) = (f[2s-1], f[2s])
@inline LTA._mma_setpair(::CUDABackendTag, f::NTuple{4,Float32}, s::Integer, x, y) =
    ntuple(i -> i == 2s - 1 ? Float32(x) : i == 2s ? Float32(y) : f[i], Val(4))

# Lane l holds rows g and g+8 (slots 1, 2), columns 2t and 2t+1; the four lanes of a row are l, l^1, l^2, l^3.
@inline LTA._mma_lane(::CUDABackendTag, lane) = (lane >> Int32(2), Int32(2) * (lane & Int32(3)))
@inline LTA._mma_rowmax(::CUDABackendTag, x) = (x = max(x, CUDACore.shfl_xor_sync(0xffffffff, x, 1));
    max(x, CUDACore.shfl_xor_sync(0xffffffff, x, 2)))
@inline LTA._mma_rowsum(::CUDABackendTag, x) = (x = x + CUDACore.shfl_xor_sync(0xffffffff, x, 1);
    x + CUDACore.shfl_xor_sync(0xffffffff, x, 2))
