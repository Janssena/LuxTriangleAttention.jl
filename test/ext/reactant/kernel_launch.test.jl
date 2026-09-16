# The SAME backward as `test/kernels/flash/backward.test.jl`, reached the OTHER way: as the reverse
# rule on a `stablehlo.custom_call`, via the Reactant extension's traced registry-entry launcher
# (`_sdpa_kernel_launch` / `_traced_reverse` in `ext/LuxTriangleAttentionReactantExt.jl`). Lives under
# `test/ext/reactant/` rather than `test/kernels/` because what is under test here is the
# EXTENSION's own machinery — the LTA-layout <-> kernel-layout reversals, the operand ORDER in the
# buffer table, the `(output, lse)` results being fed back in as the reverse's own operands, and the
# `enzyme.reverse` attribute actually landing on the call — not the kernel math itself, which the
# sibling file already pins independently.
#
# The reverse rule is called DIRECTLY here rather than through `Enzyme.gradient`, and that is the
# point: it is an ordinary function emitting ordinary custom_calls, which jax-mps executes today, so
# its numbers are checkable now. Having Enzyme *invoke* it additionally requires an Enzyme-JAX built
# with the `enzyme.reverse` custom-call rule, which the released Reactant does not ship — on stock
# Reactant an `Enzyme.gradient` through this path fails with "could not compute the adjoint for this
# operation". Splitting the two means the rule's own correctness is pinned regardless of that, and
# the only thing left resting on the unreleased pass is whether Enzyme calls it (which that pass
# verifies by signature, so it is a loud failure rather than a silent one).

using Test, Random, Reactant, Metal
include(joinpath(@__DIR__, "..", "..", "mps_client.jl"))
include(joinpath(@__DIR__, "..", "..", "ir_utils.jl"))   # reverse_symbol, reverse_signature

const _MPS = mps_client()

if _MPS === nothing
    @warn "jax-mps Metal PJRT plugin not found; skipping traced device-kernel tests" maxlog = 1
else
    # The default client is process-global, and every later test file traces against whatever it is
    # left as, so it is restored below rather than left pointing at Metal.
    const _PREV_BACKEND = Reactant.XLA.default_backend()
    Reactant.set_default_backend(_MPS)
    try

    @testset "traced device kernel (custom_call)" begin
        rng = Random.Xoshiro(42)
        T = Float32
        hd, H, N, B = 8, 4, 8, 1

        q = randn(rng, T, hd, H, N, N, B)
        k = randn(rng, T, hd, H, N, N, B)
        v = randn(rng, T, hd, H, N, N, B)
        bias = randn(rng, T, N, N, H, 1, B)
        mask = fill(true, N, 1, 1, N, B)
        mask[2, :, :, 3, :] .= false   # sparse holes only: an all-masked row softmaxes to NaN

        # `_ensure_variant` refuses to tune the `:default` kernel mid-trace (compiling it is eager
        # device work the tracer would otherwise capture — see its docstring in
        # `src/kernels/backend.jl`) and throws instead. Tuned explicitly here, ahead of any `@jit`
        # call below, rather than relying on some earlier test file having already done it as a side
        # effect — that dependency is real (`Metal`'s in-memory kernel registry persists for the rest
        # of the process either way) but fragile to lean on across files. `Metal` must already be
        # loaded for `tune_flash_kernel!` to find a backend; `test/ext/metal/*.test.jl` guarantees
        # that earlier in `test/ext/runtests.jl`.
        Metal.functional() && LuxTriangleAttention.tune_flash_kernel!(N; dim=hd, T)

        dev = Reactant.to_rarray
        qd, kd, vd, biasd, maskd = dev(q), dev(k), dev(v), dev(bias), dev(mask)

        @testset "forward parity" begin
            fo(q, k, v, m, b) = first(LuxTriangleAttention._sdpa_kernel(q, k, v, m, b))
            fl(q, k, v, m, b) = last(LuxTriangleAttention._sdpa_kernel(q, k, v, m, b))

            ref = LuxTriangleAttention._sdpa_default(q, k, v, mask, bias)
            _, ref_lse = reference_sdpa(q, k, v, mask, bias)

            @test Array(@jit fo(qd, kd, vd, maskd, biasd)) ≈ ref rtol = 1.0f-5
            @test Array(@jit fl(qd, kd, vd, maskd, biasd)) ≈ ref_lse rtol = 1.0f-5
        end

        @testset "forward parity, mask varies per batch element (B > 1)" begin
            # This entry point already builds a full `[B,Nj,Nq]` mask buffer for both branches (see
            # `_sdpa_kernel_launch`'s docstring), so this is purely a kernel-addressing regression
            # test: before `mheads` existed, `bh ÷ nh` used the WRONG stride here (`nh = H`, giving
            # `bh ÷ H` the batch index only by the coincidence that `nh` and the true head count are
            # equal for this traced path) — this test does not distinguish that coincidence from a
            # real fix, but the direct-Metal equivalent in `test/ext/metal/launch.test.jl` does,
            # since there `nh = 1` and only `mheads` can recover the batch index.
            fo2(q, k, v, m, b) = first(LuxTriangleAttention._sdpa_kernel(q, k, v, m, b))
            B2 = 2
            q2 = randn(rng, T, hd, H, N, N, B2)
            k2 = randn(rng, T, hd, H, N, N, B2)
            v2 = randn(rng, T, hd, H, N, N, B2)
            bias2 = randn(rng, T, N, N, H, 1, B2)
            mask2 = fill(true, N, 1, 1, N, B2)
            mask2[2, :, :, 3, 1] .= false
            mask2[5, :, :, 6, 2] .= false

            ref2 = LuxTriangleAttention._sdpa_default(q2, k2, v2, mask2, bias2)
            q2d, k2d, v2d, bias2d, mask2d = dev(q2), dev(k2), dev(v2), dev(bias2), dev(mask2)
            got2 = Array(@jit fo2(q2d, k2d, v2d, mask2d, bias2d))
            @test got2 ≈ ref2 rtol = 1.0f-5
            @test !(got2[:, :, :, :, 1] ≈ got2[:, :, :, :, 2])   # not vacuous
        end

        @testset "reverse rule parity" begin
            # `_sdpa_kernel_bwd!` takes and returns LuxTriangleAttention-layout arrays, converting
            # into the kernel's layout internally — so the harness hands it exactly what the forward
            # produced and compares the result directly against a host VJP. This is the same walk
            # over the same entry stages that the `enzyme.reverse` function performs; only the entry
            # point differs.
            function rev(q, k, v, mask, bias)
                o, lse = LuxTriangleAttention._sdpa_kernel(q, k, v, mask, bias)
                # dO = 1 makes this the VJP of `loss = sum(o)`.
                dO = Reactant.Ops.fill(1.0f0, size(q))
                dq, dk, dv, db = LuxTriangleAttention._sdpa_kernel_bwd!(
                    q, k, v, mask, bias, o, lse, dO)
                return (dq, dk, dv, reshape(db, size(bias)))
            end

            got = Array.(@jit rev(qd, kd, vd, maskd, biasd))

            lref(q, k, v, m, b) = sum(LuxTriangleAttention._sdpa_default(q, k, v, m, b))
            want = @jit Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(lref),
                qd, kd, vd, Enzyme.Const(maskd), biasd)

            for (i, j, nm) in ((1, 1, "dq"), (2, 2, "dk"), (3, 3, "dv"), (4, 5, "dbias"))
                @testset "$nm" begin
                    @test got[i] ≈ Array(want[j]) rtol = 1.0f-4 atol = 1.0f-5
                end
            end
        end

        @testset "rectangular forward + reverse (Nq ≠ Nj)" begin
            # MSA-row shaped: attend over Nq for each of Nj outer rows, pair bias `[Nq, Nq, H]`.
            # The traced path declares every result shape itself (`_kernel_shape`), so the `dbias`
            # shape — keys × queries, NO outer axis — is pinned here and not only on concrete arrays.
            Nq, Nj = 8, 13
            qr, kr, vr = (randn(rng, T, hd, H, Nq, Nj, B) for _ in 1:3)
            br = randn(rng, T, Nq, Nq, H, 1, B)
            mr = fill(true, Nq, 1, 1, Nj, B)
            mr[2, :, :, 3, :] .= false
            mr[Nq, :, :, Nj, :] .= false
            qrd, krd, vrd, brd, mrd = dev(qr), dev(kr), dev(vr), dev(br), dev(mr)

            fo(q, k, v, m, b) = first(LuxTriangleAttention._sdpa_kernel(q, k, v, m, b))
            @test Array(@jit fo(qrd, krd, vrd, mrd, brd)) ≈
                  LuxTriangleAttention._sdpa_default(qr, kr, vr, mr, br) rtol = 1.0f-5

            function rev(q, k, v, mask, bias)
                o, lse = LuxTriangleAttention._sdpa_kernel(q, k, v, mask, bias)
                dO = Reactant.Ops.fill(1.0f0, size(q))
                dq, dk, dv, db = LuxTriangleAttention._sdpa_kernel_bwd!(
                    q, k, v, mask, bias, o, lse, dO)
                return (dq, dk, dv, reshape(db, size(bias)))
            end
            got = Array.(@jit rev(qrd, krd, vrd, mrd, brd))
            lref(q, k, v, m, b) = sum(LuxTriangleAttention._sdpa_default(q, k, v, m, b))
            want = @jit Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(lref),
                qrd, krd, vrd, Enzyme.Const(mrd), brd)
            # `dbias` is the one to watch here: `flash_bwd_dq!` ACCUMULATES it with atomics, and
            # jax-mps allocates custom-call outputs uninitialised (a bare `allocator::malloc` in
            # metal_kernel_lib.cc). Without the `"zero_init"` flag `_emit_stage` puts on the `dbias`
            # result, `dbias` lands on top of whatever memory MLX recycled — the square case above
            # passed only when that memory happened to be fresh. See docs/BUG_traced_dbias.md.
            for (i, j, nm) in ((1, 1, "dq"), (2, 2, "dk"), (3, 3, "dv"), (4, 5, "dbias"))
                @testset "$nm" begin
                    @test got[i] ≈ Array(want[j]) rtol = 1.0f-4 atol = 1.0f-5
                end
            end
        end

        @testset "reverse rule is attached to the forward call" begin
            # The structural half: the numerics above prove the rule COMPUTES the right thing, this
            # proves the forward custom_call actually carries it, with the convention Enzyme-JAX's
            # custom-call rule checks. (`test/ext/reactant/custom_call.test.jl` pins the same
            # convention against a mock, on any backend and with no Metal.)
            fwd(q, k, v, m, b) = first(LuxTriangleAttention._sdpa_kernel(q, k, v, m, b))
            ir = repr(Reactant.@code_hlo optimize = false donated_args = :none fwd(
                qd, kd, vd, maskd, biasd))

            @test occursin("enzyme.reverse = @", ir)
            # q, k, v, bias are differentiable; the i1 mask (operand 4) is deliberately not.
            @test occursin("enzyme.active_operands = array<i64: 0, 1, 2, 3>", ir)
            # 5 primal operands + 2 primal results + 2 result cotangents -> 4 cotangents.
            @test reverse_signature(ir) == (9, 4)
            # The rule body is the three backward stages (dq+dbias, dk, dv), as three more
            # custom_calls; `>= 2` rather than pinning the exact count so an implementation change in
            # how the stages split does not make this test the first thing to break.
            sym = reverse_symbol(ir)
            body = ir[findfirst(sym, ir)[1]:end]
            @test length(collect(eachmatch(r"stablehlo\.custom_call", body))) >= 2
            # ...and the accumulated `dbias` result asks jax-mps to zero it first.
            @test occursin("zero_init", body)
        end
    end

    finally
        Reactant.set_default_backend(_PREV_BACKEND)
    end
end
