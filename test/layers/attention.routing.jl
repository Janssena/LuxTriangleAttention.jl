# `Attention`/`TriangleAttention`'s own dispatch on `chunk_size` and `use_kernel`: each keeps the
# forward output unchanged relative to neither being set, and the two are rejected together at
# construction (a fused kernel already tiles the keys, so there is nothing left for `chunk_size` to
# slice). SDPA-path-level routing (does `_sdpa_kernel` itself return the right thing) is
# `test/kernels/backend.test.jl`; this is the layer-level seam that picks a path.

@testset "use_kernel / chunk_size routing" begin
    rng = Random.Xoshiro(42)
    T = Float32
    hd, H, N, B = 8, 4, 8, 1

    x = randn(rng, T, 32, N, N, B)
    mk(; kw...) = TriangleAttention(32, hd, H; is_starting=true, kw...)
    base = mk()
    ps, st = Lux.setup(rng, base)
    ref = first(base(x, ps, st))

    @test first(mk(; chunk_size=4)(x, ps, st)) ≈ ref
    @test first(mk(; use_kernel=static(true))(x, ps, st)) ≈ ref rtol = 1.0f-5

    # `chunk_size` and `use_kernel` are alternatives: a kernel already tiles the keys, so
    # there is no full score tensor left to slice. Rejected at construction.
    @test_throws Exception mk(; chunk_size=4, use_kernel=static(true))
end
