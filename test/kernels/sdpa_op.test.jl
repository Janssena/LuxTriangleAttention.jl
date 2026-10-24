# Attention's kernels as KernelDispatch sees them: selection, the built-in kernels' variant keys and their
# fallbacks, the manifest, and how operands become a binary's arguments. No GPU and no compiled
# kernel needed: binaries are stand-in files, copied and hashed but never opened. (KernelDispatch' own
# suite covers the op-generic mechanics; `stage_args.test.jl` the exact argument lists.)

using Test, LuxTriangleAttention
import KernelDispatch
using KernelDispatch: KernelBinary
const LTA = LuxTriangleAttention
const SDPA = LTA.SDPA

# An isolated cache root, so these tests neither read nor clobber the user's real kernel cache.
const _REG_TMP = mktempdir()

function with_tmp_cache(f)
    old = get(ENV, "KERNELDISPATCH_CACHE", nothing)
    ENV["KERNELDISPATCH_CACHE"] = _REG_TMP
    # The live variant tables (`KernelDispatch.variants`) of the built-in kernels, put back afterwards.
    saved = [k => copy(KernelDispatch.variants(SDPA(), k)) for k in (LTA.FlashScalar(), LTA.FlashMMA())]
    sel = LTA.current_kernel()
    try
        f()
    finally
        for (k, vs) in saved
            live = KernelDispatch.variants(SDPA(), k)
            empty!(live)
            merge!(live, vs)
        end
        LTA.use_kernel!(sel)
        old === nothing ? delete!(ENV, "KERNELDISPATCH_CACHE") : (ENV["KERNELDISPATCH_CACHE"] = old)
    end
end

fake_bin(name) = (p = joinpath(mktempdir(), "$name.metallib"); write(p, rand(UInt8, 16)); p)
stand_ins(names) = Dict(n => KernelBinary(fake_bin(String(n)); threadgroup=32, abi=:ka, signature=false) for n in names)

key(; dtype=:Float32, dim=8, n=128, hb=true, hm=true) =
    LTA._attention_variant_key(LTA.FlashScalar(), getfield(Base, dtype), dim, n; hasbias=hb, hasmask=hm)

@testset "attention kernels" begin
    @testset "selection" begin
        # Both built-in kernels are registered by `__init__`, `:default` selected.
        names = [k.name for k in list_kernels()]
        @test [:default, :flash_mma] ⊆ names
        prev = LTA.current_kernel()
        try
            @test KernelDispatch.use_kernel!(SDPA, :flash_mma) === :flash_mma
            @test LTA.current_kernel() === :flash_mma
            @test KernelDispatch.use_kernel!(:sdpa, :default) === :default          # by op name
            @test KernelDispatch.current_kernel(SDPA) === :default
            @test use_kernel!(:flash_mma) === :flash_mma                       # this package's shorthand
            @test_throws ArgumentError KernelDispatch.use_kernel!(SDPA, :not_registered)
            @test_throws ArgumentError KernelDispatch.use_kernel!(:no_such_op, :default)
        finally
            use_kernel!(prev)
        end
    end

    @testset "variant keys and their fallbacks" begin
        op, k = SDPA(), LTA.FlashScalar()
        near(ks, q) = KernelDispatch.nearest(op, k, ks, q)
        # What `tune_flash_kernel!(lengths; dim)` registers per length: dtype, head dim, the tuned
        # length, bias/mask presence. The attention geometry (outer extent, heads) is not a key.
        @test key(n=100) == (:Float32, 8, 100, true, true)
        # A call runs the closest tuned length (absolute distance), ties to the larger.
        ln = [key(n=256), key(n=512)]
        @test near(ln, key(n=300)) == key(n=256)
        @test near(ln, key(n=400)) == key(n=512)
        @test near(ln, key(n=384)) == key(n=512)        # equidistant: the larger
        @test near(ln, key(n=64)) == key(n=256)
        @test near(ln, key(n=2000)) == key(n=512)
        @test near([key(n=256), key(n=512), key(n=768), key(n=1024)], key(n=700)) == key(n=768)
        # A different dtype, head dim or bias/mask presence is a different binary: never substituted.
        @test near(ln, key(dtype=:Float16)) === nothing
        @test near(ln, key(dim=16)) === nothing
        @test near(ln, key(hb=false)) === nothing
        # A key of the former (geometry-keyed) shape is never taken for one of the current shape.
        @test near([(:Float32, 8, 128, 128, 0, true, true)], key()) === nothing
    end

    @testset "the manifest survives a restart, unless the kernel sources changed" begin
        with_tmp_cache() do
            op, k = SDPA(), LTA.FlashScalar()
            empty!(KernelDispatch.variants(op, k))
            KernelDispatch.add_variant!(op, k, key(); stand_ins((:fwd, :dq, :dk, :dv))...)
            KernelDispatch.add_variant!(op, k, key(dim=16); stand_ins((:fwd, :dq, :dk, :dv))...)
            snapshot = copy(KernelDispatch.variants(op, k))
            @test length(snapshot) == 2
            # Binaries were copied into the cache, so they survive their originals moving.
            @test all(b -> startswith(b.file, _REG_TMP), [b for v in values(snapshot) for b in values(v)])

            empty!(KernelDispatch.variants(op, k))
            KernelDispatch.register_kernel!(op, k)
            @test KernelDispatch.variants(op, k) == snapshot

            # Built from other kernel sources: the registry holds COPIES of the binaries, so an
            # edited kernel would otherwise keep being served from it. Dropped, to be re-tuned.
            path = joinpath(KernelDispatch.cache_dir(op), "default", "manifest.toml")
            write(path, replace(read(path, String),
                "source_tag = \"$(KernelDispatch.source_tag(k))\"" => "source_tag = \"stale\""))
            empty!(KernelDispatch.variants(op, k))
            KernelDispatch.register_kernel!(op, k)
            @test isempty(KernelDispatch.variants(op, k))
        end
    end

    @testset "a bias converted to 16-bit saturates instead of overflowing" begin
        # OpenFold / Boltz fold the mask into the bias as `1e9·(mask - 1)`: `-Inf` in Float16.
        @test Float16(-1f9) == -Inf16
        lo = LTA.mask_value(Float16)
        @test LTA._to_storage(Float16, Float32[-1f9, 3, 1f9, -Inf, -2]) == Float16[lo, 3, -lo, lo, -2]
        @test eltype(LTA._to_storage(LTA.BFloat16, Float32[-1f9])) === LTA.BFloat16
        @test Float32(only(LTA._to_storage(LTA.BFloat16, Float32[-1f9]))) ≈ -1f9 rtol = 1e-2   # in range
        # Through `_sdpa_prep`: a Float32 bias beside Float16 activations.
        q = rand(Float16, 8, 2, 6, 6, 1)
        bias = fill(-1f9, 6, 6, 2, 1, 1)
        _, _, _, _, b, _ = LTA._sdpa_prep(q, q, q, nothing, bias, true)
        @test eltype(b) === Float16 && all(==(lo), b)
    end

    @testset "operands as the binaries read them" begin
        q, k, v = (rand(Float32, 8, 2, 6, 6, 1) for _ in 1:3)
        bias, mask = rand(Float32, 6, 6, 2, 1, 1), rand(Bool, 6, 1, 1, 6, 1)
        # A KEEP mask becomes a DROP mask `[Nq, Nj, B]`; the bias `[Nq, Nq, H, B]`.
        _, _, _, m, b, d = LTA._sdpa_prep(q, k, v, mask, bias, true)
        @test m == .!reshape(mask, 6, 6, 1)
        @test size(b) == (6, 6, 2, 1)
        @test (d.hasbias, d.hasmask) == (true, true)
        # Absent: a 1-element placeholder for a kernel with specialisations, else full size.
        _, _, _, m, b, d = LTA._sdpa_prep(q, k, v, nothing, nothing, true)
        @test length(b) == 1 && length(m) == 1 && !d.hasbias && !d.hasmask
        _, _, _, m, b, _ = LTA._sdpa_prep(q, k, v, nothing, nothing, false)
        @test size(b) == (6, 6, 2, 1) && iszero(b)
        @test size(m) == (6, 6, 1) && !any(m)                      # all-live
        # A plain `[hd, H, N, B]` sequence runs at rank 5 with an outer extent of one.
        q4 = rand(Float32, 8, 2, 6, 3)
        q5, = LTA._sdpa_prep(q4, q4, q4, nothing, nothing, true)
        @test size(q5) == (8, 2, 6, 1, 3)
        @test_throws ArgumentError LTA._sdpa_prep(Float64.(q), k, v, nothing, nothing, true)
    end
end

# ───────────────────────────────────────────────────────────────────────────────────────────────────
# SDPA path parity (this was `backend.test.jl`; `src/kernels/backend.jl` no longer exists)

# Forward parity of every SDPA routing path in `src/kernels/sdpa_op.jl` and `src/layers/chunk.jl`
# against `_sdpa_default` (= `Lux.scaled_dot_product_attention`).
#
# The paths exist only to keep the score tensor small — chunking slices dim 4, a device kernel tiles
# the keys inside a pre-compiled binary. None of them may change the result, so this pins them all
# to the one reference.
#
# `_chunked_sdpa` is exact (it reassociates nothing); a tiled device kernel differs by float
# reassociation from its online softmax, hence the looser tolerance where one is exercised.
#
# `SDPA()`'s return shape is unconditional — `(output, lse)` always — but `lse` is `nothing` on
# the host path, where there is no reverse rule to feed a residual to (see `SDPA`'s docstring in
# `src/kernels/sdpa_op.jl`); a plain host `Array` always resolves to `CPUBackend`, which is what
# every test here uses. `reference_sdpa` (`test/support/sdpa_reference.jl`) is an independent ground
# truth for `lse`, checked against LuxLib for `output`; a REGISTERED device kernel's own residual is
# checked in `test/kernels/launch.gpu.test.jl` and `test/ext/reactant/kernel_launch.test.jl`; the same paths traced by Reactant are
# compared with the ones here in `test/ext/reactant/sdpa_op.test.jl`.

@testset "SDPA path parity" begin
    rng = Random.Xoshiro(42)
    T = Float32
    hd, H, N, B = 8, 4, 8, 1

    q = randn(rng, T, hd, H, N, N, B)
    k = randn(rng, T, hd, H, N, N, B)
    v = randn(rng, T, hd, H, N, N, B)
    bias = randn(rng, T, N, N, H, 1, B)          # [Nk, Nq, H, 1, B]
    # Sparse holes only: a fully-masked row softmaxes over all -Inf to NaN, and NaN ≈ NaN is false.
    mask = fill(true, N, 1, 1, N, B)
    mask[2, :, :, 3, :] .= false
    mask[5, :, :, 6, :] .= false

    for (mname, m, b) in (("no mask/bias", nothing, nothing),
        ("bias only", nothing, bias),
        ("mask + bias", mask, bias))
        @testset "$mname" begin
            ref = LuxTriangleAttention._sdpa_default(q, k, v, m, b)
            ref_out, _ = reference_sdpa(q, k, v, m, b)
            @test ref_out ≈ ref   # reference_sdpa agrees with LuxLib's own output

            @test LuxTriangleAttention._chunked_sdpa(4, q, k, v, m, b) ≈ ref

            # A plain host `Array` always resolves to `CPUBackend`, so `SDPA()` runs
            # `_sdpa_default` here REGARDLESS of which kernel is selected — see `SDPA`'s docstring.
            # It returns the `(output, lse)` pair unconditionally, with
            # `lse === nothing` on this path because there is no reverse rule to feed a residual to.
            host_out, host_lse = LuxTriangleAttention.SDPA()(q, k, v, m, b)
            @test host_out ≈ ref rtol = 1.0f-5
            @test host_lse === nothing
        end
    end
end
