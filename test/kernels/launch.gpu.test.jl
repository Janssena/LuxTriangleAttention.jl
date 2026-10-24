# The CONCRETE-array device path: `SDPA()` reached with real `DeviceArray`s, no Reactant.
#
# This is the entry-driven launcher (KernelDispatch' `ext/KernelDispatchMetalExt.jl`), and it is the third way into the same
# compiled kernel — the other two being the tuner's packed `run_stage!` launches (exercised in
# `test/kernels/flash/backward.gpu.test.jl`) and the traced `custom_call`
# (`test/ext/reactant/kernel_launch.test.jl`). It is worth its own file because it is
# the only one that hands the kernel LuxTriangleAttention's own `[hd, H, Nq, Nj, B]` arrays without
# any conversion, relying entirely on the head stride (`nh = H`) to address them.
#
# There is no "decline" any more: the registry names exactly one kernel and `SDPA()` runs it or
# throws. A wrong answer is still silent from the caller's side, which is what the reference
# comparisons below are for.

using Test, Random
import KernelDispatch


if GPU !== nothing
    @testset "concrete Metal device kernel (entry-driven launcher)" begin
        rng = Random.Xoshiro(42)
        T = Float32
        hd, H, N, B = 8, 4, 8, 1

        # The launcher runs whatever `current_kernel()` names, so the built-in kernel has to be compiled
        # for this configuration first. Tuning is idempotent and cached.
        tune_flash_kernel!(N; dim=hd, T)

        q = randn(rng, T, hd, H, N, N, B)
        k = randn(rng, T, hd, H, N, N, B)
        v = randn(rng, T, hd, H, N, N, B)
        bias = randn(rng, T, N, N, H, 1, B)
        # Sparse holes only: a fully-masked row softmaxes over all -Inf to NaN.
        mask = fill(true, N, 1, 1, N, B)
        mask[2, :, :, 3, :] .= false
        mask[5, :, :, 6, :] .= false

        qd, kd, vd, biasd = DeviceArray.((q, k, v, bias))

        for (mname, m) in (("no mask", nothing), ("mask", mask))
            @testset "$mname" begin
                md = m === nothing ? nothing : DeviceArray(m)
                ref = LuxTriangleAttention._sdpa_default(q, k, v, m, bias)
                _, ref_lse = reference_sdpa(q, k, v, m, bias)

                o, lse = LuxTriangleAttention.SDPA()(qd, kd, vd, md, biasd)
                @test size(o) == size(q)
                @test size(lse) == (1, N, H, N, B)
                @test Array(o) ≈ ref rtol = 1.0f-5
                @test Array(lse) ≈ ref_lse rtol = 1.0f-5

            end
        end

        @testset "mask varies per batch element (B > 1)" begin
            # The regression test for the mask's batch stride: `flash_fwd!`'s mask address used to
            # have no batch term at all (`mb = i * n`), so with `B > 1` every batch element silently
            # read batch 0's mask. Fixed by a separate `mheads` argument (the true head count, which
            # `_emit` now passes as `H`) that makes `bh ÷ mheads` the real batch index. Two batch
            # elements here get DELIBERATELY DIFFERENT masks — mask[1] blocks a key that mask[2]
            # keeps, and vice versa — so a batch-blind kernel produces the wrong output for at least
            # one of them; a batch-correct one matches `_sdpa_default` for both.
            B2 = 2
            q2 = randn(rng, T, hd, H, N, N, B2)
            k2 = randn(rng, T, hd, H, N, N, B2)
            v2 = randn(rng, T, hd, H, N, N, B2)
            bias2 = randn(rng, T, N, N, H, 1, B2)
            mask2 = fill(true, N, 1, 1, N, B2)
            mask2[2, :, :, 3, 1] .= false   # batch 1 only
            mask2[5, :, :, 6, 2] .= false   # batch 2 only, a DIFFERENT key/row than batch 1's

            ref2 = LuxTriangleAttention._sdpa_default(q2, k2, v2, mask2, bias2)
            _, ref2_lse = reference_sdpa(q2, k2, v2, mask2, bias2)

            q2d, k2d, v2d, bias2d, mask2d = DeviceArray.((q2, k2, v2, bias2, mask2))
            o2, lse2 = LuxTriangleAttention.SDPA()(q2d, k2d, v2d, mask2d, bias2d)
            @test Array(o2) ≈ ref2 rtol = 1.0f-5
            @test Array(lse2) ≈ ref2_lse rtol = 1.0f-5

            # A caller that (wrongly) shares one batch element's mask across both would fail this:
            # confirm the two batch elements are not interchangeable, i.e. the test is not vacuous.
            @test !(Array(o2)[:, :, :, :, 1] ≈ Array(o2)[:, :, :, :, 2])
        end

        # Rectangular problems: `Nq ≠ Nj` is every attention that is not triangle attention (MSA
        # rows attend over residues for each sequence, columns the reverse). Both orderings, and a
        # batch whose elements get different masks, so `nouter` meets the batch stride too.
        @testset "rectangular Nq=$Nq Nj=$Nj B=$Br" for (Nq, Nj, Br) in ((8, 13, 1), (11, 4, 2))
            tune_flash_kernel!(Nq; dim=hd, T)
            qr, kr, vr = (randn(rng, T, hd, H, Nq, Nj, Br) for _ in 1:3)
            biasr = randn(rng, T, Nq, Nq, H, 1, Br)      # a pair bias: no Nj axis
            maskr = fill(true, Nq, 1, 1, Nj, Br)          # a key mask per outer index
            maskr[2, :, :, 3, 1] .= false
            maskr[Nq, :, :, Nj, Br] .= false

            for (mname, m) in (("mask", maskr), ("no mask", nothing))
                ref = LuxTriangleAttention._sdpa_default(qr, kr, vr, m, biasr)
                _, ref_lse = reference_sdpa(qr, kr, vr, m, biasr)
                o, lse = LuxTriangleAttention.SDPA()(DeviceArray.((qr, kr, vr))...,
                    m === nothing ? nothing : DeviceArray(m), DeviceArray(biasr))
                @test size(lse) == (1, Nq, H, Nj, Br)
                @test Array(o) ≈ ref rtol = 1.0f-5
                @test Array(lse) ≈ ref_lse rtol = 1.0f-5
            end
        end

        @testset "no bias == explicit zero bias" begin
            # Column attention has no bias; the launcher substitutes zeros rather than refusing.
            o0, _ = LuxTriangleAttention.SDPA()(qd, kd, vd, nothing, nothing)
            oz, _ = LuxTriangleAttention.SDPA()(qd, kd, vd, nothing,
                DeviceArray(zeros(T, N, N, H, 1, B)))
            @test Array(o0) ≈ Array(oz)
            @test Array(o0) ≈ LuxTriangleAttention._sdpa_default(q, k, v, nothing, nothing) rtol = 1.0f-5
        end

        @testset "rank-4 q is a plain sequence [hd, H, N, B]" begin
            # What `Attention` builds for a `[C, N, B]` input: one attention over `N`, i.e. an
            # outer extent of ONE. Rank 4 used to be refused as the packed `[hd, n, n, bh]` layout,
            # which no caller of this entry point produces; the sequence layout is the one that
            # reaches it from the layer.
            Bs = 2
            qs, ks, vs = (randn(rng, T, hd, H, N, Bs) for _ in 1:3)
            biass = randn(rng, T, N, N, H, Bs)
            masks = fill(true, N, 1, 1, Bs)
            masks[3, :, :, 2] .= false
            ref = LuxTriangleAttention._sdpa_default(qs, ks, vs, masks, biass)
            _, ref_lse = reference_sdpa(qs, ks, vs, masks, biass)
            o, lse = LuxTriangleAttention.SDPA()(DeviceArray.((qs, ks, vs, masks, biass))...)
            @test size(o) == size(qs)
            @test size(lse) == (1, N, H, Bs)
            @test Array(o) ≈ ref rtol = 1.0f-5
            @test Array(lse) ≈ ref_lse rtol = 1.0f-5
        end

        @testset "views of device memory reach the kernel (fused q/k/v)" begin
            # `Attention(...; fuse_qkv=true)` hands the kernel `view`s of one projection. These used
            # to read as HOST arrays (`backend_of(::SubArray) = CPUBackend()`), so the layer silently
            # ran LuxLib attention with `use_kernel=true` — correct numbers, no kernel.
            qkv = DeviceArray(randn(rng, T, hd, H, 3, N, N, B))
            qv, kv, vv = (view(qkv, :, :, i, :, :, :) for i in 1:3)
            @test LuxTriangleAttention.backend_of(qv) isa typeof(backend_tag())
            o, lse = LuxTriangleAttention.SDPA()(qv, kv, vv, nothing, biasd)
            od, _ = LuxTriangleAttention.SDPA()(copy(qv), copy(kv), copy(vv), nothing, biasd)
            @test Array(o) == Array(od)
            @test Array(o) ≈ LuxTriangleAttention._sdpa_default(Array(qv), Array(kv), Array(vv), nothing, bias) rtol = 1.0f-5

            # …and through the layer: same parameters, kernel vs LuxLib. Not bitwise equal is the
            # point: equal outputs would mean the kernel did not run.
            C = hd * H
            plain = Attention(C, hd, H)
            fused = Attention(C, hd, H; use_kernel=LuxTriangleAttention.static(true))
            ps, st = Lux.setup(rng, plain)
            x = randn(rng, T, C, N, N, B)
            psd = Lux.fmap(a -> a isa AbstractArray ? DeviceArray(a) : a, ps)
            y_ref = first(plain(x, ps, st))
            y_ker = Array(first(fused(DeviceArray(x), psd, st)))
            @test y_ker ≈ y_ref rtol = 1.0f-5
            @test y_ker != y_ref
        end
    end

    @testset "$(nameof(T)) storage == Float32 kernel on upcast inputs, :$(KernelDispatch.kernelname(kern))" for T in (Float16, LuxTriangleAttention.BFloat16),
            kern in (LuxTriangleAttention.FlashScalar(), LuxTriangleAttention.FlashMMA())
        # Accumulation is Float32 whatever the storage type, so the only difference from the Float32
        # kernel is input/output rounding (8-bit mantissa for BFloat16) and, for `:flash_mma`, the
        # 16-bit operands the tuner may choose. Output and gradient buffers follow the storage type;
        # `delta` is reduced in Float32 (the kernels bind it as such), and `dbias` comes back in the
        # bias's own precision. See docs/done/BUG_traced_kernel_16bit_results.md.
        relL2(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))
        tolf, tolb = T === Float16 ? (1e-3, 1e-2) : (1e-2, 3e-2)
        rng = Random.Xoshiro(9)
        hd, H, Nq, Nj, B = 16, 4, 12, 7, 1
        tune = kern isa LuxTriangleAttention.FlashScalar ? tune_flash_kernel! : tune_flash_mma!
        tune(Nq; dim=hd, T)
        tune(Nq; dim=hd, T=Float32)
        r(dims...) = T.(randn(rng, Float32, dims...))      # `randn` has no BFloat16 method
        q, k, v, dO = (r(hd, H, Nq, Nj, B) for _ in 1:4)
        bias = r(Nq, Nq, H, 1, B)
        mask = fill(true, Nq, 1, 1, Nj, B)
        mask[2, :, :, 3, :] .= false
        md = DeviceArray(mask)
        h, s = DeviceArray.((q, k, v, bias, dO)), DeviceArray.(Float32.(x) for x in (q, k, v, bias, dO))
        op = LuxTriangleAttention.SDPA()
        fwd(x) = KernelDispatch.forward(op, kern, x[1], x[2], x[3], md, x[4])
        bwd(x, o, lse) = KernelDispatch.backward(op, kern, (x[1], x[2], x[3], md, x[4]), (o, lse), (x[5], nothing))

        o16, lse16 = fwd(h)
        o32, lse32 = fwd(s)
        @test eltype(o16) === T
        @test eltype(lse16) === Float32
        @test relL2(Array(o16), Array(o32)) < tolf
        @test relL2(Array(lse16), Array(lse32)) < tolf

        g16 = bwd(h, o16, lse16)
        g32 = bwd(s, o32, lse32)
        @test eltype.((g16[1], g16[2], g16[3], g16[5])) == (T, T, T, T)
        for (a, b) in zip((g16[1], g16[2], g16[3], g16[5]), (g32[1], g32[2], g32[3], g32[5]))
            @test relL2(Array(a), Array(b)) < tolb
        end

        # A Float32 bias beside 16-bit activations is converted, not reinterpreted.
        o_mixed, _ = KernelDispatch.forward(op, kern, h[1], h[2], h[3], md, s[4])
        @test Array(o_mixed) == Array(o16)
    end
    @testset "a mask folded into a Float32 bias, Float16 activations, :$(KernelDispatch.kernelname(kern))" for kern in
            (LuxTriangleAttention.FlashScalar(), LuxTriangleAttention.FlashMMA())
        # `1e9·(mask - 1)` (OpenFold, Boltz) is `-Inf` in Float16; `_sdpa_prep` saturates the converted
        # bias at `mask_value(Float16)` instead. Partially masked queries must then equal the Float32
        # kernel with the -1e9 bias; a query masked on EVERY key through the bias stays finite (it attends
        # by softmax(q·k), not uniformly — such rows are padding).
        relL2(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))
        rng = Random.Xoshiro(11)
        hd, H, Nq, Nj, B = 16, 2, 12, 5, 1
        tune = kern isa LuxTriangleAttention.FlashScalar ? tune_flash_kernel! : tune_flash_mma!
        tune(Nq; dim=hd, T=Float16)
        tune(Nq; dim=hd, T=Float32)
        q, k, v, dO = (randn(rng, Float16, hd, H, Nq, Nj, B) for _ in 1:4)
        keep = trues(Nq, Nq)                       # [key, query]
        keep[[2, 4], 5] .= false                   # query 5: two keys masked
        keep[:, 3] .= false                        # query 3: every key masked
        bias = randn(rng, Float32, Nq, Nq, H, 1, B) .+ 1f9 .* (Float32.(keep) .- 1)
        op = LuxTriangleAttention.SDPA()
        db = DeviceArray(bias)
        h, s = DeviceArray.((q, k, v, dO)), DeviceArray.(Float32.(x) for x in (q, k, v, dO))

        o16, lse16 = KernelDispatch.forward(op, kern, h[1], h[2], h[3], nothing, db)
        o32, _ = KernelDispatch.forward(op, kern, s[1], s[2], s[3], nothing, db)
        o16h, o32h = Array(o16), Array(o32)
        @test all(isfinite, o16h) && all(isfinite, Array(lse16))
        live = [j for j in 1:Nq if j != 3]
        @test relL2(o16h[:, :, live, :, :], o32h[:, :, live, :, :]) < 1e-3

        g = KernelDispatch.backward(op, kern, (h[1], h[2], h[3], nothing, db), (o16, lse16), (h[4], nothing))
        @test all(x -> all(isfinite, Array(x)), (g[1], g[2], g[3], g[5]))
        @test eltype(g[5]) === Float32                # `dbias` in the bias's own precision
    end

    @testset "binary signatures and mixed-precision operands, :$(KernelDispatch.kernelname(kern))" for kern in
            (LuxTriangleAttention.FlashScalar(), LuxTriangleAttention.FlashMMA())
        # KernelDispatch ≥ 0.2: a binary built here records the types it was compiled for, a launch binding
        # other types is refused (naming the argument), and the op converts a `k`, `v` or `dO` of another
        # precision to `q`'s before it binds anything.
        LTA = LuxTriangleAttention
        rng = Random.Xoshiro(21)
        hd, H, Nq, Nj, B = 16, 2, 12, 5, 1
        tune = kern isa LTA.FlashScalar ? tune_flash_kernel! : tune_flash_mma!
        for T in (Float16, Float32)
            tune(Nq; dim=hd, T)
        end
        op = LTA.SDPA()
        r(T, dims...) = DeviceArray(T.(randn(rng, Float32, dims...)))
        q16, k16, v16, dO16 = (r(Float16, hd, H, Nq, Nj, B) for _ in 1:4)
        bias = r(Float32, Nq, Nq, H, 1, B)

        @testset "every variant records its signature" begin
            vs = KernelDispatch.variants(op, kern)
            @test !isempty(vs)
            for (key, stages) in vs, (stage, b) in stages
                @test b.signature isa Tuple
                tok = key[1] === :Float16 ? "f16" : key[1] === :Float32 ? "f32" : "bf16"
                @test b.signature[1] == tok                                       # the leading output
            end
        end

        @testset "a launch with other types than the binary was compiled for is refused" begin
            _, _, _, m, b, d = LTA._sdpa_prep(q16, k16, v16, nothing, bias, true)
            q32, k32, v32 = (Float32.(x) for x in (q16, k16, v16))
            a = (; o=KernelDispatch.OutArray(Float32, LTA._full(d)), lse=KernelDispatch.OutArray(Float32, LTA._res(d)),
                q=q32, k=k32, v=v32, bias=Float32.(b), bias_t=Float32.(LTA._pair_t(b, d)), mask=m)
            args = LTA.stage_args(kern, Val(:fwd), a, LTA._scalars(d))
            b16 = KernelDispatch.variants(op, kern)[LTA._key(d)][:fwd]
            err = try
                KernelDispatch.prepare_launch(b16, args...; extent=LTA._extent(d))
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("f16", err.msg) && occursin("f32", err.msg)
        end

        @testset "k, v and dO of another precision are converted" begin
            k32, v32, dO32 = (Float32.(x) for x in (k16, v16, dO16))   # exactly the Float16 values
            o16, lse16 = KernelDispatch.forward(op, kern, q16, k16, v16, nothing, bias)
            om, lsem = KernelDispatch.forward(op, kern, q16, k32, v32, nothing, bias)
            @test eltype(om) === Float16
            @test Array(om) == Array(o16) && Array(lsem) == Array(lse16)
            g16 = KernelDispatch.backward(op, kern, (q16, k16, v16, nothing, bias), (o16, lse16), (dO16, nothing))
            gm = KernelDispatch.backward(op, kern, (q16, k32, v32, nothing, bias), (o16, lse16), (dO32, nothing))
            @test eltype(gm[1]) === Float16 && eltype(gm[2]) === Float32 && eltype(gm[3]) === Float32
            @test Array(gm[1]) == Array(g16[1])
            @test Array(gm[2]) == Float32.(Array(g16[2])) && Array(gm[3]) == Float32.(Array(g16[3]))
        end
    end
end
