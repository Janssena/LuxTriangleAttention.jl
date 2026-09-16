# Forward parity of every SDPA routing path in `src/kernels/backend.jl` and `src/layers/chunk.jl`
# against `_sdpa_default` (= `Lux.scaled_dot_product_attention`).
#
# The paths exist only to keep the score tensor small — chunking slices dim 4, a device kernel tiles
# the keys inside a pre-compiled binary. None of them may change the result, so this pins them all
# to the one reference.
#
# `_chunked_sdpa` is exact (it reassociates nothing); a tiled device kernel differs by float
# reassociation from its online softmax, hence the looser tolerance where one is exercised.
#
# `_sdpa_kernel`'s return shape is unconditional — `(output, lse)` always — but `lse` is `nothing` on
# the host path, where there is no reverse rule to feed a residual to (see its docstring in
# `src/kernels/backend.jl`); a plain host `Array` always resolves to `CPUBackend`, which is what
# every test here uses. `reference_sdpa` (`test/kernels/sdpa_reference.jl`) is an independent ground
# truth for `lse`, checked against LuxLib for `output`; a REGISTERED device kernel's own residual is
# checked in `test/ext/metal/launch.test.jl` and `test/ext/reactant/kernel_launch.test.jl`.

using Reactant

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

            # A plain host `Array` always resolves to `CPUBackend`, so `_sdpa_kernel` runs
            # `_sdpa_default` here REGARDLESS of which kernel is selected — see its docstring in
            # `src/kernels/backend.jl`. It returns the `(output, lse)` pair unconditionally, with
            # `lse === nothing` on this path because there is no reverse rule to feed a residual to.
            host_out, host_lse = LuxTriangleAttention._sdpa_kernel(q, k, v, m, b)
            @test host_out ≈ ref rtol = 1.0f-5
            @test host_lse === nothing

            @testset "under Reactant" begin
                dev = Reactant.to_rarray
                qd, kd, vd = dev(q), dev(k), dev(v)
                md, bd_ = m === nothing ? nothing : dev(m), b === nothing ? nothing : dev(b)
                run(f) = Array((@compile f(qd, kd, vd, md, bd_))(qd, kd, vd, md, bd_))

                @test run((a, c, d, e, f) -> LuxTriangleAttention._sdpa_default(a, c, d, e, f)) ≈ ref
                @test run((a, c, d, e, f) -> LuxTriangleAttention._chunked_sdpa(4, a, c, d, e, f)) ≈ ref

                # Same reasoning as the native case above: a CPU client means `backend_of` answers
                # `CPUBackend`, so `_sdpa_kernel` traces to exactly `_sdpa_default`.
                @test run((a, c, d, e, f) -> first(LuxTriangleAttention._sdpa_kernel(a, c, d, e, f))) ≈
                      ref rtol = 1.0f-5
            end
        end
    end
end
