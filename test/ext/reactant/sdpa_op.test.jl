# `SDPA` and the chunked SDPA traced by Reactant (its default, CPU client), against the same Float32 paths run natively on the
# host: `test/kernels/sdpa_op.test.jl`. A CPU client means `backend_of` answers `CPUBackend`, so `SDPA()` traces to exactly
# `_sdpa_default`; the kernels in a trace are `kernel_launch.test.jl`.

using Reactant

@testset "SDPA paths, traced" begin
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
            ref = LuxTriangleAttention._sdpa_default(q, k, v, m, b)          # the native layer's path, on the host
            dev = Reactant.to_rarray
            qd, kd, vd = dev(q), dev(k), dev(v)
            md, bd_ = m === nothing ? nothing : dev(m), b === nothing ? nothing : dev(b)
            run(f) = (args = (qd, kd, vd, md, bd_); Array(Reactant.compile(f, args)(args...)))

            @test run((a, c, d, e, f) -> LuxTriangleAttention._sdpa_default(a, c, d, e, f)) ≈ ref
            @test run((a, c, d, e, f) -> LuxTriangleAttention._chunked_sdpa(4, a, c, d, e, f)) ≈ ref

            # Same reasoning as the native case above: a CPU client means `backend_of` answers
            # `CPUBackend`, so `SDPA()` traces to exactly `_sdpa_default`.
            @test run((a, c, d, e, f) -> first(LuxTriangleAttention.SDPA()(a, c, d, e, f))) ≈
                  ref rtol = 1.0f-5
        end
    end
end
