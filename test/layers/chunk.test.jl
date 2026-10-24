# Chunked attention: `Attention(...; chunk_size=n)` splits the attention over dim 4 of the prepared
# q/k/v — the first axis it batches over, since dim 3 is the attended token axis. For a pair input
# `[C, Ni, Nj, B]` that is `Nj`, a TOKEN axis, so this bites at B=1; for a 3-D input it is the
# sample batch and is a no-op at B=1.
#
# Why it exists: for a pair input `[C, Ni, Nj, B]` the attention scores are `[Nq, Nk, H, Nj, B]`,
# i.e. O(N³·H). At N=400 that is ~1 GiB per attention op, which makes a reverse pass over a single
# pairformer block exceed 32 GB. Chunking bounds the live score tensor to `chunk_size·N²·H`.
# (The reference does the same — openfold/boltz `chunk_size_tri_attn`.)
#
# These tests pin the property that matters: chunking must be a pure memory/compute rearrangement
# and change NOTHING about the result, for every combination of rank, mask and bias — including
# chunk sizes that do not divide the axis, and chunk sizes larger than it.

@testset "Chunked attention" begin
    rng = Random.Xoshiro(42)

    @testset "4-D pair input (:qk, as TriangleAttention)" begin
        chn, N, B = 32, 7, 2          # N=7 is prime, so most chunk sizes are non-divisors
        head_dim, num_heads = 8, 4

        for T in (Float32, Float64)
            x = randn(rng, T, chn, N, N, B)
            bias = randn(rng, T, num_heads, N, N, B)
            # Sparse holes, not a fully random mask: a fully-masked row softmaxes over all -Inf to
            # NaN, and `NaN ≈ NaN` is false — the comparison would then fail on identical outputs.
            mask = fill(true, N, N, B); mask[2, 3, :] .= false; mask[5, 6, :] .= false

            cfgs = (
                ("no mask, no bias", nothing, nothing),
                ("mask only", mask, nothing),
                ("bias only", nothing, bias),
                ("mask + bias", mask, bias),
            )

            for (name, m, b) in cfgs
                @testset "$name ($T)" begin
                    ref_layer = Attention(chn, head_dim, num_heads; bias_layout=:qk)
                    ps, st = Lux.setup(rng, ref_layer) |> convert_types(T)
                    y_ref, _ = ref_layer(x, b, m, ps, st)

                    # 1 and N are the boundaries; 3 and 4 do not divide 7; N+3 exceeds the axis.
                    for cs in (1, 3, 4, N, N + 3)
                        layer = Attention(chn, head_dim, num_heads; bias_layout=:qk, chunk_size=cs)
                        y, _ = layer(x, b, m, ps, st)      # identical ps: same function
                        @test y ≈ y_ref
                        @test size(y) == size(y_ref)
                    end
                end
            end

            @testset "type-stability ($T)" begin
                layer = Attention(chn, head_dim, num_heads; bias_layout=:qk, chunk_size=3)
                ps, st = Lux.setup(rng, layer) |> convert_types(T)
                @test_nowarn @inferred layer(x, bias, mask, ps, st)
            end
        end
    end

    @testset "3-D input (:kq) chunks over the batch axis" begin
        chn, N, B = 32, 6, 4
        head_dim, num_heads = 8, 4
        T = Float32
        x = randn(rng, T, chn, N, B)
        bias = randn(rng, T, num_heads, N, N, B)
        mask = fill(true, N, B); mask[3, :] .= false

        ref_layer = Attention(chn, head_dim, num_heads; bias_layout=:kq)
        ps, st = Lux.setup(rng, ref_layer) |> convert_types(T)
        y_ref, _ = ref_layer(x, bias, mask, ps, st)

        for cs in (1, 3, B, B + 2)
            layer = Attention(chn, head_dim, num_heads; bias_layout=:kq, chunk_size=cs)
            y, _ = layer(x, bias, mask, ps, st)
            @test y ≈ y_ref
        end
    end

    @testset "TriangleAttention forwards chunk_size" begin
        chn, N, B = 32, 7, 2
        head_dim, num_heads = 8, 4
        T = Float32
        x = randn(rng, T, chn, N, N, B)
        mask = fill(true, N, N, B); mask[2, 3, :] .= false; mask[5, 6, :] .= false

        for is_starting in (true, false), (mname, m) in (("no mask", nothing), ("mask", mask))
            @testset "$(is_starting ? "starting" : "ending") / $mname" begin
                ref = TriangleAttention(chn, head_dim, num_heads; is_starting)
                ps, st = Lux.setup(rng, ref) |> convert_types(T)
                y_ref, _ = ref(x, m, ps, st)

                for cs in (1, 3, N)
                    l = TriangleAttention(chn, head_dim, num_heads; is_starting, chunk_size=cs)
                    y, _ = l(x, m, ps, st)
                    @test y ≈ y_ref
                end
            end
        end
    end

    # Regression: this dispatch called an undefined `_triangle_attention_forward` and always threw.
    @testset "TriangleAttention NamedTuple dispatch" begin
        chn, N, B = 32, 6, 1
        T = Float32
        x = randn(rng, T, chn, N, N, B)
        mask = fill(true, N, N, B); mask[2, 3, :] .= false
        l = TriangleAttention(chn, 8, 4; is_starting=true)
        ps, st = Lux.setup(rng, l) |> convert_types(T)
        y_pos, _ = l(x, mask, ps, st)
        y_nt, _ = l((; x, mask), ps, st)
        @test y_nt ≈ y_pos
        @test first(l((; x), ps, st)) ≈ first(l(x, nothing, ps, st))   # mask optional
    end

    @testset "invalid chunk_size is rejected at construction" begin
        @test_throws AssertionError Attention(32, 8, 4; chunk_size=0)
        @test_throws AssertionError Attention(32, 8, 4; chunk_size=-1)
    end
end

# `_sdpa_enzyme_safe` no longer has a base-package method — it is now local to the Reactant extension
# (nothing in `src/` ever called it; see `ext/LuxTriangleAttentionReactantExt/chunk.jl`). Its parity check moved with it:
# `test/ext/reactant/chunk.test.jl`.
