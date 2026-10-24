# The fragment lane layout `:flash_mma`'s in-register softmax depends on (ext/LuxTriangleAttentionMetalExt/flash_mma.jl).
#
# Apple does not document which entries of an 8×8 `simdgroup_matrix` each lane holds; the kernel
# relies on the layout measured on an M1 Pro (and used by MLX): two adjacent entries of one row per
# lane, the four lanes of a row differing in lane bits 0 and 3. If a chip or OS changed it, the
# kernel would mix rows in its softmax and return wrong numbers without any error — so it is pinned
# here, through the very hooks the kernel calls.

using Test, Metal
using LuxTriangleAttention.KernelAbstractions
import KernelDispatch
using KernelDispatch: OutArray, run_binary

@kernel unsafe_indices = true function _lane_probe!(pairs, rc, red, unused)
    be = KernelDispatch.MetalBackendTag()
    l = Int32(@index(Local, Linear)) - Int32(1)
    ms = @localmem Float32 (8, 8)
    for e in (l, l + Int32(32))
        r, c = e % Int32(8), e ÷ Int32(8)
        @inbounds ms[r+Int32(1), c+Int32(1)] = Float32(8 * r + c)
    end
    @synchronize
    f = LuxTriangleAttention._mma_load(be, ms, 1, 1)
    x, y = LuxTriangleAttention._mma_pair(be, f)
    row, col = LuxTriangleAttention._mma_lane(be, l)
    unsafe_store!(pairs, x, 2l + Int32(1)); unsafe_store!(pairs, y, 2l + Int32(2))
    unsafe_store!(rc, Float32(row), 2l + Int32(1)); unsafe_store!(rc, Float32(col), 2l + Int32(2))
    # Row reductions over this lane's own entries: max and sum of the whole row.
    unsafe_store!(red, LuxTriangleAttention._mma_rowmax(be, max(x, y)), 2l + Int32(1))
    unsafe_store!(red, LuxTriangleAttention._mma_rowsum(be, x + y), 2l + Int32(2))
    # setpair round trip: the fragment stored back must hold the replaced entries.
    g = LuxTriangleAttention._mma_setpair(be, f, -x, -y)
    @synchronize
    Metal.simdgroup_store(g, ms)
    @synchronize
    for e in (l, l + Int32(32))
        unsafe_store!(pairs, @inbounds(ms[e % Int32(8) + Int32(1), e ÷ Int32(8) + Int32(1)]), Int32(65) + e)
    end
end

# The probe needs no op or registry: compile the KA kernel and `run_binary` it. One simdgroup, one group.
if Metal.functional()
    @testset "flash_mma fragment lane layout" begin
        # The binary is only rebuilt when absent: the name carries the source's hash.
        bin = KernelDispatch.ka_compile(KernelDispatch.MetalBackendTag(), _lane_probe!,
            ntuple(_ -> AbstractArray{Float32}, 4); tg=32,
            name="lane_probe_" * string(hash(read(@__FILE__)); base=36))
        pairs, rc, red = run_binary(bin, OutArray(Float32, 128), OutArray(Float32, 64),
            OutArray(Float32, 64), MtlArray(zeros(Float32, 1)); extent=32)   # 32 lanes: one group
        P, RC, RED = Array(pairs), Int.(Array(rc)), Array(red)
        M = [Float32(8r + c) for r in 0:7, c in 0:7]
        seen = falses(8, 8)
        for l in 0:31
            row, col = RC[2l+1], RC[2l+2]
            # the lane holds (row, col) and (row, col+1) …
            @test P[2l+1] == M[row+1, col+1]
            @test P[2l+2] == M[row+1, col+2]
            seen[row+1, col+1] = seen[row+1, col+2] = true
            # … and the row reductions see the whole row
            @test RED[2l+1] == maximum(M[row+1, :])
            @test RED[2l+2] == sum(M[row+1, :])
        end
        @test all(seen)                                  # every entry held by exactly one lane pair
        @test P[65:128] == vec(-M)                       # setpair replaced every lane's entries
    end
end
