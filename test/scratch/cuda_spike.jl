# Phase 0 of the CUDA port (docs: the plan in the PR / CHANGELOG): four questions a CUDA machine has to answer
# before the fragment hooks are generalised. Nothing here is tested automatically; read the printed report.
#
#   julia --project=test test/scratch/cuda_spike.jl
#
# 1. Do the scalar `:default` stages compile through `KernelDispatch.ka_compile` for CUDA? (It REFUSES a
#    kernel whose compiled code can still throw: `Int32(x)` of an Int64, `%` / `÷` by a runtime value.)
# 2. Does inline PTX `mma.sync.m16n8k16` work from Julia, and is the register layout the one the hook
#    design assumes (A: rows g, g+8 x cols 2t, 2t+1, 2t+8, 2t+9;  B: rows 2t, 2t+1, 2t+8, 2t+9 x col g;
#    C: rows g, g+8 x cols 2t, 2t+1, with g = lane >> 2, t = lane & 3)? Float16 and BFloat16.
# 3. Does `CUDA.atomic_add!` on a raw `LLVMPtr{Float32,AS.Global}` compile and add inside a KA kernel?
# 4. How much static shared memory does a KA `@localmem` kernel get? (The cap is 48 KiB per block.)

using CUDA
using CUDA: CUDACore
using KernelAbstractions
using KernelDispatch
using LuxTriangleAttention
using BFloat16s: BFloat16

const LTA = LuxTriangleAttention
const KD = KernelDispatch
const Interop = CUDACore.LLVM.Interop

results = Pair{String,Any}[]
note(name, ok, detail="") = (push!(results, name => (ok, detail)); println(ok ? "  ok    " : "  FAIL  ", name, isempty(detail) ? "" : "  — " * first(split(detail, '\n')));)

dev = CUDA.device()
cap = CUDA.capability(dev)
println("device: ", CUDA.name(dev), "  sm_", cap.major, cap.minor, "  CUDA ", CUDA.runtime_version(), "  Julia ", VERSION)
println("max threads/block: ", CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
    "   shared/block (default): ", CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
    "   shared/block (opt-in): ", CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN))
println("KernelDispatch device backend: ", KD.device_backend(), "   available: ", KD.device_available())

# --- 1. the scalar stages --------------------------------------------------------------------------
println("\n[1] ka_compile of the :default stages")
let caps = LTA.device_caps(KD.device_backend()), k = LTA.FlashScalar()
    println("  device_caps: ", caps)
    for T in (Float32, Float16, BFloat16), stage in LTA.FLASH_STAGES, (hb, hm) in ((true, true), (false, false))
        grid = LTA.tune_grid(k, stage, 32, T, caps)
        p = map(first, grid)
        try
            b = LTA.build_stage(k, stage, T, 32, p; hasbias=hb, hasmask=hm)
            note("$T :$stage hb=$hb hm=$hm $(p)", true, "max threads of the binary: $(KD.max_threads(b))")
        catch err
            note("$T :$stage hb=$hb hm=$hm $(p)", false, sprint(showerror, err))
            println("          ", first(split(sprint(showerror, err), '\n'), 1)...)
        end
    end
end

# --- 2. mma.sync m16n8k16 --------------------------------------------------------------------------
println("\n[2] inline-PTX mma.sync.m16n8k16 (f32 accumulate)")

@inline mma_f16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3) = Interop.@asmcall(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {\$0,\$1,\$2,\$3}, {\$4,\$5,\$6,\$7}, {\$8,\$9}, {\$10,\$11,\$12,\$13};",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", true, NTuple{4,Float32},
    Tuple{UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,Float32,Float32,Float32,Float32},
    a0, a1, a2, a3, b0, b1, c0, c1, c2, c3)
@inline mma_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3) = Interop.@asmcall(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {\$0,\$1,\$2,\$3}, {\$4,\$5,\$6,\$7}, {\$8,\$9}, {\$10,\$11,\$12,\$13};",
    "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", true, NTuple{4,Float32},
    Tuple{UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,Float32,Float32,Float32,Float32},
    a0, a1, a2, a3, b0, b1, c0, c1, c2, c3)

# Two adjacent 16-bit values as one 32-bit register (the lower index in the low half).
@inline pack2(x::UInt16, y::UInt16) = UInt32(x) | (UInt32(y) << 16)
@inline bits16(x::Float16) = reinterpret(UInt16, x)
@inline bits16(x::BFloat16) = reinterpret(UInt16, x)

# One warp. `a` is 16x16 row-major, `b` is 16x8 stored column-major (b[k + 16n]), `c` is 16x8 row-major (c[8m + n]).
# Each lane loads its registers from the layout the design assumes; if the product comes out right, the layout is right.
# `unsafe_trunc` rather than `Int32(...)`: a conversion that can throw is refused at build time on CUDA.
@kernel unsafe_indices = true function mma_probe!(c, a, b, ::Val{T}) where {T}
    lane = unsafe_trunc(Int32, @index(Local, Linear)) - Int32(1)
    g = lane >> Int32(2)
    t = lane & Int32(3)
    A(r, k) = bits16(unsafe_load(a, r * Int32(16) + k + Int32(1)))
    B(k, n) = bits16(unsafe_load(b, k + Int32(16) * n + Int32(1)))
    a0 = pack2(A(g, 2t), A(g, 2t + Int32(1)))
    a1 = pack2(A(g + Int32(8), 2t), A(g + Int32(8), 2t + Int32(1)))
    a2 = pack2(A(g, 2t + Int32(8)), A(g, 2t + Int32(9)))
    a3 = pack2(A(g + Int32(8), 2t + Int32(8)), A(g + Int32(8), 2t + Int32(9)))
    b0 = pack2(B(2t, g), B(2t + Int32(1), g))
    b1 = pack2(B(2t + Int32(8), g), B(2t + Int32(9), g))
    d = T === Float16 ? mma_f16(a0, a1, a2, a3, b0, b1, 0f0, 0f0, 0f0, 0f0) :
        mma_bf16(a0, a1, a2, a3, b0, b1, 0f0, 0f0, 0f0, 0f0)
    unsafe_store!(c, d[1], g * Int32(8) + 2t + Int32(1))
    unsafe_store!(c, d[2], g * Int32(8) + 2t + Int32(2))
    unsafe_store!(c, d[3], (g + Int32(8)) * Int32(8) + 2t + Int32(1))
    unsafe_store!(c, d[4], (g + Int32(8)) * Int32(8) + 2t + Int32(2))
end

for T in (Float16, BFloat16)
    try
        am = rand(Float32, 16, 16) .- 0.5f0               # a[r+1, k+1]
        bm = rand(Float32, 16, 8) .- 0.5f0                # b[k+1, n+1]
        a_dev = CuArray(T.(permutedims(am)))              # row-major: element (r, k) at r*16 + k
        b_dev = CuArray(T.(bm))                           # column-major: (k, n) at k + 16n
        c_dev = CUDA.zeros(Float32, 8, 16)                # row-major 16x8 seen as 8x16 column-major
        bin = KD.ka_compile(KD.device_backend(), mma_probe!,
            (AbstractArray{Float32}, AbstractArray{T}, AbstractArray{T}, Val{T}); tg=32,
            name="lta_spike_mma_$(T)", dir=mktempdir())
        KD.run_binary(bin, c_dev, a_dev, b_dev; extent=(32, 1, 1))
        got = permutedims(Array(c_dev))                   # 16x8
        want = Float32.(T.(am)) * Float32.(T.(bm))
        err = maximum(abs, got .- want)
        note("mma.sync m16n8k16 $T layout (max |err| = $(round(err; sigdigits=3)))", err < 1e-3)
    catch err
        note("mma.sync m16n8k16 $T", false, sprint(showerror, err))
        showerror(stdout, err); println()
    end
end

# --- 3. atomic add on a raw pointer ----------------------------------------------------------------
println("\n[3] CUDA.atomic_add! on a raw global Float32 pointer")
@inline fadd!(ptr::Core.LLVMPtr{Float32,CUDACore.AS.Global}, idx, val::Float32) =
    CUDA.atomic_add!(ptr + (Int(idx) - 1) * sizeof(Float32), val)
@kernel unsafe_indices = true function atomic_probe!(out)
    # `out` is bound as a raw device pointer (an `LLVMPtr{Float32,AS.Global}`), as the flash kernels' `dbias` is.
    fadd!(out, Int32(1), 1f0)
end
try
    bin = KD.ka_compile(KD.device_backend(), atomic_probe!, (AbstractArray{Float32},); tg=64,
        name="lta_spike_atomic", dir=mktempdir())
    out = CUDA.zeros(Float32, 4)
    KD.run_binary(bin, out; extent=(128, 1, 1))
    note("atomic_add! (expect out[1] == 128, got $(Array(out)[1]))", Array(out)[1] == 128f0)
catch err
    note("atomic_add! on a raw pointer", false, sprint(showerror, err))
    showerror(stdout, err); println()
end

# --- 4. shared memory ------------------------------------------------------------------------------
println("\n[4] static @localmem size")
@kernel unsafe_indices = true function shmem_probe!(out, ::Val{NEL}) where {NEL}
    s = @localmem Float32 (NEL,)
    i = unsafe_trunc(Int32, @index(Local, Linear))
    @inbounds s[i] = 1f0
    @synchronize
    unsafe_store!(out, @inbounds(s[i]), i)
end
for kib in (32, 48, 64)
    try
        KD.ka_compile(KD.device_backend(), shmem_probe!, (AbstractArray{Float32}, Val{kib * 1024 ÷ 4});
            tg=32, name="lta_spike_shmem_$kib", dir=mktempdir())
        note("@localmem of $kib KiB compiles", true)
    catch err
        note("@localmem of $kib KiB compiles", false, sprint(showerror, err))
    end
end

println("\nsummary: ", count(r -> last(r)[1], results), " ok, ", count(r -> !last(r)[1], results), " failed")
println("Paste everything above back; failures in [1] name the construct the CUDA build refuses.")
