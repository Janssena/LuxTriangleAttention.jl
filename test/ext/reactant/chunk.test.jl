# Chunked attention UNDER `Reactant.@compile`, where the extension replaces the unrolled Julia loop
# with a rolled `@trace` / `stablehlo.while`.
#
# Why the rolled form exists: inside a trace the plain loop is unrolled, and the chunks are mutually
# independent, so XLA schedules them concurrently and keeps every slice's scores live — measured to
# give no memory win at all. `stablehlo.while` forces one slice at a time.
#
# What is pinned here is correctness: rolling must not change the result, which is the native Julia layer's (the
# traced Float32 layer is compared with it, chunked and not). Chunk sizes that do not divide the axis must still work
# (they fall back to the unrolled form).

using Reactant, Enzyme

@testset "Chunked attention (Reactant)" begin
    rng = Random.Xoshiro(42)
    T = Float32
    chn, N, B = 32, 8, 1                      # N=8 so 2/4/8 divide it and 3/5 do not
    head_dim, num_heads = 8, 4
    dev = Reactant.to_rarray

    x = randn(rng, T, chn, N, N, B)
    bias = randn(rng, T, num_heads, N, N, B)
    # Sparse holes only: a fully-masked row softmaxes over all -Inf to NaN, and `NaN ≈ NaN` is false.
    mask = fill(true, N, N, B); mask[2, 3, :] .= false; mask[5, 6, :] .= false

    for (mname, m, b) in (("no mask/bias", nothing, nothing),
        ("mask + bias", mask, bias))
        @testset "$mname" begin
            ref_layer = Attention(chn, head_dim, num_heads; bias_layout=:qk)
            ps, st = Lux.setup(rng, ref_layer) |> convert_types(T)

            x_d, ps_d, st_d = dev(x), dev(ps), dev(st)
            b_d = isnothing(b) ? nothing : dev(b)
            m_d = isnothing(m) ? nothing : dev(m)

            f(l, x, b, m, ps, st) = first(l(x, b, m, ps, st))
            cf_ref = @compile f(ref_layer, x_d, b_d, m_d, ps_d, st_d)
            y_ref = Array(cf_ref(ref_layer, x_d, b_d, m_d, ps_d, st_d))
            y_native = first(ref_layer(x, b, m, ps, st))                # the native layer, on the host
            @test y_ref ≈ y_native

            # 2 and 4 divide N (rolled path); 3 and 5 do not (unrolled fallback); N disables it.
            for cs in (2, 3, 4, 5, N)
                l = Attention(chn, head_dim, num_heads; bias_layout=:qk, chunk_size=cs)
                cf = @compile f(l, x_d, b_d, m_d, ps_d, st_d)
                y = Array(cf(l, x_d, b_d, m_d, ps_d, st_d))
                @test y ≈ y_ref
                @test y ≈ y_native
            end
        end
    end

    # Correctness above says nothing about whether the loop stayed rolled — an unrolled or
    # batch-fissioned module gives identical numbers and zero memory benefit, which is exactly how
    # the no-op went unnoticed. Pin the structure by counting ops in the optimized module.
    @testset "rolled loop survives optimization" begin
        l_ref = Attention(chn, head_dim, num_heads; bias_layout=:qk)
        ps, st = Lux.setup(rng, l_ref) |> convert_types(T)
        x_d, ps_d, st_d = dev(x), dev(ps), dev(st)
        f(l, x, ps, st) = first(l(x, nothing, nothing, ps, st))

        nexp(mod) = count(_ -> true, eachmatch(r"stablehlo\.exponential", string(mod)))
        opts = LuxTriangleAttention.chunked_compile_options()

        # One softmax per attention: the unchunked graph has exactly one, and so must a rolled
        # loop (one body). An unrolled loop has `nchunks`.
        mod_ref = @code_hlo f(l_ref, x_d, ps_d, st_d)
        @test !LuxTriangleAttention.check_rolled(mod_ref)
        @test nexp(mod_ref) == 1

        # nchunks = 4, at the default unroll threshold; nchunks = 2 is under it. Both must stay
        # rolled once the threshold is dropped, so this also pins the threshold's meaning.
        old_thr = Reactant.Compiler.WHILE_UNROLL_THRESHOLD[]
        try
            Reactant.Compiler.WHILE_UNROLL_THRESHOLD[] = 0
            for cs in (2, 4)
                l = Attention(chn, head_dim, num_heads; bias_layout=:qk, chunk_size=cs)
                mod = @code_hlo compile_options = opts f(l, x_d, ps_d, st_d)
                @test LuxTriangleAttention.check_rolled(mod)
                @test nexp(mod) == 1
            end

            # Without the options the `while` is gone — the state that made chunking a no-op.
            l = Attention(chn, head_dim, num_heads; bias_layout=:qk, chunk_size=4)
            @test !LuxTriangleAttention.check_rolled(@code_hlo f(l, x_d, ps_d, st_d))
        finally
            Reactant.Compiler.WHILE_UNROLL_THRESHOLD[] = old_thr
        end
    end

    # The point of rolling is to bound the REVERSE, so the reverse has to compile and agree with the
    # unchunked gradient. It did not before `_sdpa_enzyme_safe`: Enzyme's reverse of `NNlib.softmax`'s
    # max-reduce inside a `stablehlo.while` fails MLIR verification.
    @testset "rolled reverse matches unchunked" begin
        vz = randn(rng, T, chn, N, N, B)
        loss(l, x, m, ps, st, v) = sum(first(l(x, nothing, m, ps, st)) .* v)
        gfn(l, x, m, ps, st, v) = Enzyme.gradient(
            Reverse, loss, Const(l), x, Const(m), ps, Const(st), Const(v))

        l_ref = Attention(chn, head_dim, num_heads; bias_layout=:qk)
        ps, st = Lux.setup(rng, l_ref) |> convert_types(T)
        x_d, m_d, ps_d, st_d, v_d = dev(x), dev(mask), dev(ps), dev(st), dev(vz)

        cg_ref = @compile gfn(l_ref, x_d, m_d, ps_d, st_d, v_d)
        g_ref = Array(cg_ref(l_ref, x_d, m_d, ps_d, st_d, v_d)[2])

        old_thr = Reactant.Compiler.WHILE_UNROLL_THRESHOLD[]
        try
            Reactant.Compiler.WHILE_UNROLL_THRESHOLD[] = 0
            opts = LuxTriangleAttention.chunked_compile_options()
            # This works only because the rolled path routes through the extension's `_sdpa_enzyme_safe`
            # shadow of LuxLib's ReactantExt SDPA, whose softmax max is `ignore_derivatives`-wrapped.
            # With LuxLib's own softmax the reverse fails MLIR verification outright.
            l = Attention(chn, head_dim, num_heads; bias_layout=:qk, chunk_size=2)
            cg = @compile compile_options = opts gfn(l, x_d, m_d, ps_d, st_d, v_d)
            g = Array(cg(l, x_d, m_d, ps_d, st_d, v_d)[2])
            @test all(isfinite, g)
            @test g ≈ g_ref rtol = 1e-4

            # NOT tested here, and known broken: chunk_size=4 (nchunks=2) aborts the reverse with
            # "update rank does not match operand rank: 6 vs 5" on the chunk write's
            # `dynamic_update_slice`. It is an `LLVM ERROR` abort, not a Julia exception, so it
            # cannot be wrapped in `@test_broken` — it would take the whole test process down.
            # nchunks-dependent, so it is not the softmax: the forward is fine at both chunk sizes
            # and the reverse is fine at nchunks=4. Unresolved — do not assume a given chunk count
            # differentiates until it has been tried.
        finally
            Reactant.Compiler.WHILE_UNROLL_THRESHOLD[] = old_thr
        end
    end

    @testset "TriangleAttention under @compile" begin
        ref = TriangleAttention(chn, head_dim, num_heads; is_starting=true)
        ps, st = Lux.setup(rng, ref) |> convert_types(T)
        x_d, m_d, ps_d, st_d = dev(x), dev(mask), dev(ps), dev(st)

        g(l, x, m, ps, st) = first(l(x, m, ps, st))
        cf_ref = @compile g(ref, x_d, m_d, ps_d, st_d)
        y_ref = Array(cf_ref(ref, x_d, m_d, ps_d, st_d))
        y_native = first(ref(x, mask, ps, st))
        @test y_ref ≈ y_native

        for cs in (2, 4)
            l = TriangleAttention(chn, head_dim, num_heads; is_starting=true, chunk_size=cs)
            cf = @compile g(l, x_d, m_d, ps_d, st_d)
            y = Array(cf(l, x_d, m_d, ps_d, st_d))
            @test y ≈ y_ref
            @test y ≈ y_native
        end
    end
end

# `_sdpa_enzyme_safe` is `_sdpa_default` with the softmax written out so the max-subtraction can be marked
# non-differentiable — see its docstring in `ext/LuxTriangleAttentionReactantExt/chunk.jl`. It is defined ONLY in this
# extension (nothing in the base package calls it), so this pins the equivalence the rolled loop
# above depends on: the traced, differentiation-safe rewrite must compute the same VALUES as ordinary
# attention, on `AnyTracedRArray` — the only type it has a method for.
const RX = Base.get_extension(LuxTriangleAttention, :LuxTriangleAttentionReactantExt)

@testset "_sdpa_enzyme_safe matches _sdpa_default" begin
    rng = Random.Xoshiro(7)
    dev = Reactant.to_rarray
    for T in (Float64, Float32), rank in (4, 5)
        E, H, Nq, Nj, B = 8, 4, 8, 4, 1
        sz = rank == 4 ? (E, H, Nq, B) : (E, H, Nq, Nj, B)
        q, k, v = randn(rng, T, sz...), randn(rng, T, sz...), randn(rng, T, sz...)
        a = LuxTriangleAttention._sdpa_default(q, k, v, nothing, nothing)

        qd, kd, vd = dev(q), dev(k), dev(v)
        f(q, k, v) = RX._sdpa_enzyme_safe(q, k, v, nothing, nothing)
        b = Array(@jit f(qd, kd, vd))

        @test size(a) == size(b)
        @test a ≈ b
    end
end

# The rolled path (`_sdpa_enzyme_safe`) in 16 bit, compiled. It scales q explicitly and runs `dot_general` in
# the input type, so the same logit overflow as `test/layers/attention.test.jl`'s applies, and its
# masked logits are `typemin` (-Inf). Written before the fixes: a failure is a finding.
@testset "chunked attention, 16-bit (compiled)" begin
    dev = Reactant.to_rarray
    f(l, x, b, m, ps, st) = first(l(x, b, m, ps, st))

    @testset "logits beyond Float16 before the 1/√d scale, chunk_size=$cs" for cs in (4, 8)
        # A PAIR input: chunking slices dim 4 (`Nj`), which a `[C, N, B]` sequence does not have. N = 8:
        # chunk_size 4 rolls, 8 is the unchunked fallback.
        c = f16_overflow_attention(Random.Xoshiro(1); pair=true)
        l = Attention(16, 16, 2; chunk_size=cs)
        y64, _ = l(c.x64, c.ps64, c.st64)
        cf = @compile f(l, dev(c.x16), nothing, nothing, dev(c.ps16), dev(c.st16))
        y16 = Array(cf(l, dev(c.x16), nothing, nothing, dev(c.ps16), dev(c.st16)))
        if cs == 4
            # The rolled path (`_sdpa_enzyme_safe`): runs a Float16 chunk in Float32.
            @test all(isfinite, y16)
            @test relerr(y16, y64) < 2.0e-2
        else
            # `chunk_size >= N`: nothing to chunk, so this is LuxLib's own traced SDPA (the unchunked
            # layer), which is not Float16-safe under Reactant and is not meant for production (it
            # materialises every score). Pinned as broken so a LuxLib fix shows up.
            @test_broken all(isfinite, y16)
        end
    end

    @testset "fully masked key set is finite, $T, chunk_size=4" for T in (Float32, Float16)
        rng = Random.Xoshiro(2)
        chn, N, B = 16, 8, 1
        l = Attention(chn, 8, 2; bias_layout=:qk, chunk_size=4)
        ps, st = Lux.setup(rng, l) |> convert_types(T)
        x = T.(randn(rng, Float32, chn, N, N, B))
        mask = fill(true, N, N, B); mask[:, 3, :] .= false
        cf = @compile f(l, dev(x), nothing, dev(mask), dev(ps), dev(st))
        y = Array(cf(l, dev(x), nothing, dev(mask), dev(ps), dev(st)))
        @test all(isfinite, y)
    end
end
