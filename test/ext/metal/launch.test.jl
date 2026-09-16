# The CONCRETE-array device path: `_sdpa_kernel` reached with real `MtlArray`s, no Reactant.
#
# This is the entry-driven launcher in `ext/metal/launch.jl`, and it is the third way into the same
# compiled kernel — the other two being the direct `run_fwd!` launcher (exercised in
# `test/kernels/flash/backward.test.jl`) and the traced `custom_call`
# (`test/ext/reactant/kernel_launch.test.jl`). It is worth its own file because it is
# the only one that hands the kernel LuxTriangleAttention's own `[hd, H, Nq, Nj, B]` arrays without
# any conversion, relying entirely on the head stride (`nh = H`) to address them.
#
# There is no "decline" any more: the registry names exactly one kernel and `_sdpa_kernel` runs it or
# throws. A wrong answer is still silent from the caller's side, which is what the reference
# comparisons below are for.

using Test, Random, Metal

const LTAMx = Base.get_extension(LuxTriangleAttention, :LuxTriangleAttentionMetalExt)

if Metal.functional()
    @testset "concrete Metal device kernel (entry-driven launcher)" begin
        rng = Random.Xoshiro(42)
        T = Float32
        hd, H, N, B = 8, 4, 8, 1

        # The launcher runs whatever `LTA_KERNEL[]` names, so the built-in kernel has to be compiled
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

        qd, kd, vd, biasd = MtlArray.((q, k, v, bias))

        for (mname, m) in (("no mask", nothing), ("mask", mask))
            @testset "$mname" begin
                md = m === nothing ? nothing : MtlArray(m)
                ref = LuxTriangleAttention._sdpa_default(q, k, v, m, bias)
                _, ref_lse = reference_sdpa(q, k, v, m, bias)

                o, lse = LuxTriangleAttention._sdpa_kernel(qd, kd, vd, md, biasd)
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

            q2d, k2d, v2d, bias2d, mask2d = MtlArray.((q2, k2, v2, bias2, mask2))
            o2, lse2 = LuxTriangleAttention._sdpa_kernel(q2d, k2d, v2d, mask2d, bias2d)
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
                o, lse = LuxTriangleAttention._sdpa_kernel(MtlArray.((qr, kr, vr))...,
                    m === nothing ? nothing : MtlArray(m), MtlArray(biasr))
                @test size(lse) == (1, Nq, H, Nj, Br)
                @test Array(o) ≈ ref rtol = 1.0f-5
                @test Array(lse) ≈ ref_lse rtol = 1.0f-5
            end
        end

        @testset "no bias == explicit zero bias" begin
            # Column attention has no bias; the launcher substitutes zeros rather than refusing.
            o0, _ = LuxTriangleAttention._sdpa_kernel(qd, kd, vd, nothing, nothing)
            oz, _ = LuxTriangleAttention._sdpa_kernel(qd, kd, vd, nothing,
                MtlArray(zeros(T, N, N, H, 1, B)))
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
            o, lse = LuxTriangleAttention._sdpa_kernel(MtlArray.((qs, ks, vs, masks, biass))...)
            @test size(o) == size(qs)
            @test size(lse) == (1, N, H, Bs)
            @test Array(o) ≈ ref rtol = 1.0f-5
            @test Array(lse) ≈ ref_lse rtol = 1.0f-5
        end
    end
end
