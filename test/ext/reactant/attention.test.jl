# `Attention` compiled by Reactant at 16 bits, against the native layer, for the cases where the compiler's rewrites matter.
# The eager counterparts are `test/layers/attention.test.jl`. Written before the fixes: a failure is a finding.
#
# Under Reactant the compiler moves LuxLib's `1/√d` scale after `Q·Kᵀ` (legal in exact arithmetic), so the unscaled
# logits must fit Float16; the chunked path is Float16-safe and is `chunk.test.jl`.

using Reactant

@testset "Attention, 16-bit, compiled" begin
    dev = Reactant.to_rarray
    f(l, x, b, m, ps, st) = first(l(x, b, m, ps, st))

    @testset "logits beyond Float16 before the 1/√d scale" begin
        c = f16_overflow_attention(Random.Xoshiro(1))
        # The case is only meaningful inside this window.
        @test c.maxlogit > floatmax(Float16)
        @test c.maxscaled < floatmax(Float16) / 1.5
        @testset "compiled (LuxLib's SDPA: known not Float16-safe)" begin
            # Under Reactant the compiler moves LuxLib's `1/√d` scale after `Q·Kᵀ`, so the UNSCALED logits
            # must fit Float16 and overflow here. Not fixed on purpose: this path materialises every score
            # and is not meant for production; use `chunk_size` (Float16-safe, see
            # `test/ext/reactant/chunk.test.jl`) or `use_kernel`. Pinned as broken so a LuxLib fix shows up.
            args = (c.layer, dev(c.x16), nothing, nothing, dev(c.ps16), dev(c.st16))
            y16 = Array(Reactant.compile(f, args)(args...))
            @test eltype(y16) === Float16
            @test_broken all(isfinite, y16)
        end
    end

    @testset "raw `_sdpa_default`, compiled: q·k = 96000 before the scale, 13856 after" begin
        hd, H, N = 48, 2, 4
        q = zeros(Float16, hd, H, N, 1); k = zeros(Float16, hd, H, N, 1); v = ones(Float16, hd, H, N, 1)
        q[:, 1, 1, 1] .= 50; k[:, 1, 2, 1] .= 40
        g(q, k, v) = LuxTriangleAttention._sdpa_default(q, k, v, nothing, nothing)
        @test all(g(q, k, v) .== 1)              # eager: v ≡ 1, so every output is 1 if the weights are finite
        args = (dev(q), dev(k), dev(v))
        y = Array(Reactant.compile(g, args)(args...))
        # Compiled, LuxLib's SDPA overflows here (see the compiled testset above): known, pinned as broken.
        @test_broken count(!isfinite, y) == 0
    end

    @testset "fully masked key set is finite, $T" for T in (Float32, Float16)
        # Pair input [C, Ni, Nj, B], mask [Ni, Nj, B]: column j = 3 masks every key.
        rng = Random.Xoshiro(2)
        chn, N, B = 16, 8, 1
        l = Attention(chn, 8, 2; bias_layout=:qk)
        ps, st = Lux.setup(rng, l) |> convert_types(T)
        x = T.(randn(rng, Float32, chn, N, N, B))
        mask = fill(true, N, N, B); mask[:, 3, :] .= false
        ok = setdiff(1:N, 3)

        args = (l, dev(x), nothing, dev(mask), dev(ps), dev(st))
        y = Array(Reactant.compile(f, args)(args...))
        @test all(isfinite, y)
        # The other columns must not change: compare to the same layer with column 3 left unmasked.
        mask_open = copy(mask); mask_open[:, 3, :] .= true
        y_open, _ = l(x, nothing, mask_open, ps, st)
        @test relerr(y[:, :, ok, :], y_open[:, :, ok, :]) < (T === Float16 ? 1.0e-2 : 1.0e-5)
    end
end
