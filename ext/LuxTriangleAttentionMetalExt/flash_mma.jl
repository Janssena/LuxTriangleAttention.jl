# Metal's half of `:flash_mma` (src/kernels/flash_mma/): the fragment hooks on 8×8
# `simdgroup_matrix`es, dispatched on KernelDispatch' `MetalBackendTag`. Building, verifying, timing and
# tuning the stages is backend-agnostic (`src/kernels/launch.jl`, `src/kernels/tune.jl`).

using KernelDispatch: MetalBackendTag
using Metal: simdgroup_load, simdgroup_multiply_accumulate, simdgroup_matrix_init_filled,
    simd_shuffle_xor

const _F16x64 = NTuple{64,VecElement{Float16}}
const _BF16x64 = NTuple{64,VecElement{Metal.BFloat16}}
const _F32x64 = NTuple{64,VecElement{Float32}}

@inline LTA._mma_load(::MetalBackendTag, a, r, c) = simdgroup_load(a, (Int(r), Int(c)))
@inline LTA._mma_zero(::MetalBackendTag, ::Type{E}) where {E} = simdgroup_matrix_init_filled(zero(E))
# BFloat16 fragments from REGISTER values (the P / dS operands at `ops = BFloat16`): NOT supported by
# Apple's compiler for the M1 GPU (AGXMetalG13X). It only handles a bfloat fragment that comes from a
# matrix intrinsic (load / multiply / init_filled); one assembled in registers stores ZEROS, and one
# built by inserting runtime bfloat values crashes the compiler service at pipeline creation
# (`XPC_ERROR_CONNECTION_INTERRUPTED`). Metal.jl's own GEMM documents the same (`src/gemm.jl`). Float16
# and Float32 fragments are fine either way. So BFloat16 operands are built here as `i16` lanes with one
# bitcast: on the M1 that compiles quickly and computes zeros, the tuner's verification rejects it, and
# BFloat16 storage runs with Float32 operands. A device whose compiler accepts it gets the real thing.
const _U16x64 = NTuple{64,VecElement{UInt16}}
@inline _bf16_vec(u::_U16x64) = Base.llvmcall(
    "%r = bitcast <64 x i16> %0 to <64 x bfloat>\nret <64 x bfloat> %r", _BF16x64, Tuple{_U16x64}, u)
@inline LTA._mma_zero(::MetalBackendTag, ::Type{Metal.BFloat16}) = _bf16_vec(ntuple(_ -> VecElement(0x0000), Val(64)))
@inline function LTA._mma_frag(::MetalBackendTag, ::Type{Metal.BFloat16}, x, y)
    bx, by = reinterpret(UInt16, Metal.BFloat16(x)), reinterpret(UInt16, Metal.BFloat16(y))
    return _bf16_vec(ntuple(i -> VecElement(i == 1 ? bx : i == 2 ? by : 0x0000), Val(64)))
end
@inline LTA._mma(::MetalBackendTag, a::_F32x64, b::_F32x64, c::_F32x64) = simdgroup_multiply_accumulate(a, b, c)
# Float16 × Float16 + Float32. Metal.jl wraps only the same-type form; the AIR intrinsic exists for
# mixed operands (measured on an M1 Pro, eager and as a custom call).
@inline LTA._mma(::MetalBackendTag, a::_F16x64, b::_F16x64, c::_F32x64) = ccall(
    "extern air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32",
    llvmcall, _F32x64, (_F16x64, _F16x64, _F32x64), a, b, c)
# BFloat16 × BFloat16 + Float32, the same way (measured on an M1 Pro; native or emulated not known).
@inline LTA._mma(::MetalBackendTag, a::_BF16x64, b::_BF16x64, c::_F32x64) = ccall(
    "extern air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64bf16.v64bf16.v64f32",
    llvmcall, _F32x64, (_BF16x64, _BF16x64, _F32x64), a, b, c)

# A lane's two entries are elements 1 and 2 of its 64-wide vector; see the layout below.
@inline LTA._mma_pair(::MetalBackendTag, f) = (Float32(f[1].value), Float32(f[2].value))
@inline LTA._mma_setpair(::MetalBackendTag, f::NTuple{64,VecElement{E}}, x, y) where {E} =
    ntuple(i -> i == 1 ? VecElement{E}(E(x)) : i == 2 ? VecElement{E}(E(y)) : f[i], Val(64))

# Lane `l` (bits b4…b0) holds (row, col) and (row, col+1) with row = 4·b4 + 2·b2 + b1 and
# col = 4·b3 + 2·b0 — measured (test/ext/metal/flash_mma.test.jl), and the mapping MLX's
# kernels rely on. The four lanes of a row differ in bits 0 and 3, hence the xor masks 1 and 8.
@inline LTA._mma_lane(::MetalBackendTag, lane) =
    (Int32(4) * ((lane >> 4) & Int32(1)) + Int32(2) * ((lane >> 2) & Int32(1)) + ((lane >> 1) & Int32(1)),
     Int32(4) * ((lane >> 3) & Int32(1)) + Int32(2) * (lane & Int32(1)))
@inline LTA._mma_rowmax(::MetalBackendTag, x) = (x = max(x, simd_shuffle_xor(x, 1)); max(x, simd_shuffle_xor(x, 8)))
@inline LTA._mma_rowsum(::MetalBackendTag, x) = (x = x + simd_shuffle_xor(x, 1); x + simd_shuffle_xor(x, 8))
