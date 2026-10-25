# The forward of `src/kernels/flash_mma/forward.jl`. The kernel is run through its KernelDispatch method
# directly (see `flash_mma/runtests.jl`, which also holds the shared setup), against
# `reference_sdpa`: untiled, independent of every kernel. Float16 is compared against it at Float64 on
# the same (upcast) values (likewise BFloat16), so only the kernel's own rounding is measured.

@testset "$T, $(shape), bias=$hb, mask=$hm" for (T, shape, hb, hm) in MMA_CASES
    c = mma_case(T, shape, hb, hm)
    o, lse = mma_forward(c, T, shape)
    @test eltype(o) === T
    @test eltype(lse) === Float32

    f64(x) = x === nothing ? nothing : Float64.(x)
    ref_o, ref_lse = reference_sdpa(f64(c.q), f64(c.k), f64(c.v), c.mask, f64(c.bias))
    oh = Array(o)
    # A fully masked slice (`mma_case` has one when `hm`) attends uniformly: `o = mean(v)`, `lse = -Inf`.
    @test all(isfinite, oh)
    @test relL2_mma(oh, ref_o) < tol_fwd(T)
    @test (vec(Array(lse)) .== -Inf) == (vec(ref_lse) .== -Inf)
    @test hm == any(==(-Inf), Array(lse))
    fin = isfinite.(vec(ref_lse))
    @test relL2_mma(vec(Array(lse))[fin], vec(ref_lse)[fin]) < tol_fwd(T)
end

# `use_kernel!(:flash_mma)` must reach this kernel: through the registry the result is the one the
# direct call gave, bit for bit. (`:default` computes the same quantity to a different rounding, so a
# selection that did nothing cannot pass.)
@testset "selection routes to the kernel" begin
    T, shape = first(MMA_TYPES), (32, 2, 37, 5, 2)      # Float32 where the device runs it on its matrix units
    c = mma_case(T, shape, true, true)
    o, lse = mma_forward(c, T, shape)
    use_kernel!(:flash_mma)
    @test LTA_.current_kernel() === :flash_mma
    o2, lse2 = LTA_.SDPA()(c.d(c.q), c.d(c.k), c.d(c.v), c.d(c.mask), c.d(c.bias))
    @test isequal(Array(o2), Array(o))
    @test isequal(Array(lse2), Array(lse))
    use_kernel!(:default)
    od, _ = LTA_.SDPA()(c.d(c.q), c.d(c.k), c.d(c.v), c.d(c.mask), c.d(c.bias))
    @test !isequal(Array(od), Array(o))     # the other kernel really is a different computation
end

# Float16 whose logits leave the type's range BEFORE the 1/√d scale (> 65504), after it inside. The
# kernel is believed safe (Float32 accumulation, scaling inside); this pins it. Run directly, as above.
@testset "Float16 logits beyond range before scaling" begin
    shape = (16, 2, 16, 4, 1)
    hd, H, n, nouter, B = shape
    q, k, v = f16_range_qkv(shape...)
    @test hd * Float32(q[1])^2 > floatmax(Float16)
    c = (; q, k, v, dO=q, bias=nothing, mask=nothing, d=x -> x === nothing ? nothing : DeviceArray(x))
    o, _ = mma_forward(c, Float16, shape)
    o_ref, _ = reference_sdpa(Float32.(q), Float32.(k), Float32.(v), nothing, nothing)
    oh = Array(o)
    @test all(isfinite, oh)
    @test relL2_mma(oh, o_ref) < 2e-2
end

# KernelDispatch builds a `ka` binary's launch context from a geometry probe; it must be exactly the context
# `:flash_mma`'s own kernels get from KernelAbstractions (see `flash/forward.test.jl`).
@testset "launch context: KernelDispatch' probe == the kernel's own" begin
    KA = LuxTriangleAttention.KernelAbstractions
    tag = backend_tag()
    for stage in LTA_.FLASH_MMA_STAGES, tg in (64, 128, 256)
        groups = (cld(100, 8 * (tg ÷ 32)), 5, 3)
        obj = LTA_.flash_mma_kernel(stage)(ka_backend(), (tg, 1, 1))
        nd, _, iterspace, _ = KA.launch_config(obj, (groups[1] * tg, groups[2], groups[3]), (tg, 1, 1))
        ctx = KA.mkcontext(obj, nd, iterspace)
        pctx, pghost = KernelDispatch.ka_context(tag, groups, tg)
        @test typeof(pctx) === typeof(ctx) && pctx == ctx
        @test pghost == (sizeof(typeof(ctx)) == 0)
    end
end

# A storage type the device's matrix units do not take runs at the type they do (CUDA: Float32 at Float16): the inputs
# are rounded to it, the results come back in the caller's type. Accuracy is that of the narrower type, and the outputs are
# finite even for logits and biases beyond its range (the cast saturates).
@testset "$T storage computed at $(LTA_.mma_compute_type(LTA_.mma_backend(), T))" for T in CAST_TYPES
    Tc = LTA_.mma_compute_type(LTA_.mma_backend(), T)
    shape = (32, 2, 37, 5, 2)
    c = mma_case(T, shape, true, true)
    o, lse = mma_forward(c, T, shape)
    @test eltype(o) === T                       # the caller's type, not the narrower one
    @test eltype(lse) === Float32
    f64(x) = x === nothing ? nothing : Float64.(x)
    ref_o, ref_lse = reference_sdpa(f64(c.q), f64(c.k), f64(c.v), c.mask, f64(c.bias))
    @test all(isfinite, Array(o))
    @test relL2_mma(Array(o), ref_o) < tol_fwd(Tc) * 4      # inputs rounded to `Tc` as well as the output
    fin = isfinite.(vec(ref_lse))
    @test relL2_mma(vec(Array(lse))[fin], vec(ref_lse)[fin]) < tol_fwd(Tc) * 4
    # A bias far beyond the narrower type's range (a mask written as -1e9) stays a mask, not a NaN.
    big = merge(c, (; bias=fill(T(-1e9), size(c.bias)), mask=nothing))
    ob, _ = mma_forward(big, T, shape)
    @test all(isfinite, Array(ob))
end
