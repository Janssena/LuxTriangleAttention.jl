# The launch context of `src/kernels/flash/forward.jl`'s kernels, against KernelDispatch' (the geometry itself
# is the call's `extent` over the binary's `tile`, `(cld(n, rows), nj, H·B)`), and its 16-bit storage: Float16 and
# BFloat16 operands against `reference_sdpa` at Float64. The forward's Float32 numerics are covered through the registry
# by `test/kernels/sdpa_op.test.jl` and, against an independent loop reference, by `flash/backward.test.jl`.
#
# Metal supplies the backend object a KernelAbstractions context is built from.

using Test, Random
import KernelDispatch

if GPU !== nothing
    LTA_ = LuxTriangleAttention
    tag = backend_tag()

    # KernelDispatch builds the launch context of every KA binary from a geometry PROBE, an empty kernel, not
    # from the kernel itself (which is gone by launch time when a binary is loaded from a manifest). That
    # is only sound if the probe's context is exactly the one the kernel's own KernelAbstractions
    # launch builds: the same type (the ABI of the context slot) and the same value.
    own_context(kern, groups, tg) = begin
        obj = kern(ka_backend(), (tg, 1, 1))
        ndrange = (groups[1] * tg, groups[2], groups[3])
        nd, _, iterspace, _ = LuxTriangleAttention.KernelAbstractions.launch_config(obj, ndrange, (tg, 1, 1))
        ctx = LuxTriangleAttention.KernelAbstractions.mkcontext(obj, nd, iterspace)
        (ctx, sizeof(typeof(ctx)) == 0)
    end

    @testset "launch context: KernelDispatch' probe == the kernel's own" begin
        for stage in LTA_.FLASH_STAGES, (tg, rows) in ((128, 128), (128, 64), (256, 32)), nouter in (5, 1)
            groups = (cld(100, rows), nouter, 3)
            ctx, ghost = own_context(LTA_.stage_kernel(stage), groups, tg)
            pctx, pghost = KernelDispatch.ka_context(tag, groups, tg)
            @test typeof(pctx) === typeof(ctx)
            @test pctx == ctx
            @test pghost == ghost
            @test KernelDispatch.ka_grid(tag, groups) == (prod(groups), 1, 1)    # one flat dispatch
        end
    end

    # Float16 whose logits leave the type's range BEFORE the 1/√d scale (> 65504), after it inside.
    # The kernel is believed safe — Float32 accumulation, scaling inside — and this pins it: a kernel
    # that multiplied q·k in Float16 first would return Inf/NaN here. Concrete `DeviceArray` path via the
    # registry's `:default` kernel; reference is Float32 on the same values.
    @testset "Float16 logits beyond range before scaling" begin
        hd, H, n, B = 8, 2, 8, 1
        tune_flash_kernel!(n; dim=hd, T=Float16)
        q, k, v = f16_range_qkv(hd, H, n, n, B)
        @test hd * Float32(q[1])^2 > floatmax(Float16)
        o_ref, _ = reference_sdpa(Float32.(q), Float32.(k), Float32.(v), nothing, nothing)

        o, _ = KernelDispatch.forward(LTA_.SDPA(), LTA_.FlashScalar(), DeviceArray(q), DeviceArray(k), DeviceArray(v),
            nothing, nothing)
        o = Array(o)
        @test eltype(o) === Float16
        @test all(isfinite, o)
        @test Float32.(o) ≈ o_ref rtol = 2.0e-2
    end

    # Mixed precision: the operands and the output are stored in 16 bits, everything the kernel computes with them
    # (scores, the online softmax, the weighted sum, `lse`) in Float32. The reference is Float64 on the SAME (rounded)
    # values, so what is measured is the kernel's arithmetic and its one rounding of the output, not the input's.
    # BFloat16 has Float32's range and an 8-bit mantissa: the kernel is not asked to compute IN it.
    relL2(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))
    @testset "$(nameof(T)) storage, Float32 accumulation, bias=$hb mask=$hm" for T in (Float16, LuxTriangleAttention.BFloat16),
            hb in (true, false), hm in (true, false)

        rng = Random.Xoshiro(3)
        hd, H, n, nj, B = 16, 2, 24, 5, 2
        tune_flash_kernel!(n; dim=hd, T)
        r(dims...) = T.(randn(rng, Float32, dims...))          # `randn` has no BFloat16 method
        q, k, v = (r(hd, H, n, nj, B) for _ in 1:3)
        bias = hb ? r(n, n, H, 1, B) : nothing
        mask = hm ? rand(rng, n, 1, 1, nj, B) .> 0.2 : nothing
        hm && (mask[:, 1, 1, 2, 1] .= false)                    # one fully masked slice: it attends uniformly
        f64(x) = x === nothing ? nothing : Float64.(x)
        o_ref, lse_ref = reference_sdpa(f64(q), f64(k), f64(v), mask, f64(bias))

        d(x) = x === nothing ? nothing : DeviceArray(x)
        o, lse = KernelDispatch.forward(LTA_.SDPA(), LTA_.FlashScalar(), d(q), d(k), d(v), d(mask), d(bias))
        @test eltype(o) === T
        @test eltype(lse) === Float32
        @test all(isfinite, Array(o))
        @test relL2(Array(o), o_ref) < (T === Float16 ? 2.0e-3 : 1.0e-2)
        @test (vec(Array(lse)) .== -Inf) == (vec(lse_ref) .== -Inf)      # a fully masked slice has lse = -Inf
        @test hm == any(==(-Inf), Array(lse))
        fin = isfinite.(vec(lse_ref))
        @test relL2(vec(Array(lse))[fin], vec(lse_ref)[fin]) < 1.0e-4
    end
end
