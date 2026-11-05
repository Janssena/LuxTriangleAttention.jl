# `TriMul`'s `:default` kernel on the machine's GPU, against the layer's own forward on the same device: every
# direction, mask shape, batch size, parameter layout and bias setting it covers, a problem of several token blocks
# with a tail, and a channel count whose LayerNorm tile has to shrink to fit local memory. Then the calls it declines
# (they must run the layer's forward, not fail). Backend-agnostic: only the adapter's names (`DeviceArray`).
#
# Not here: the Enzyme reverse rule end to end (CUDA only, `test/ext/cuda/`) and tracing (eager only for now).
# 16-bit data: Float16 and BFloat16, in every contraction mode.

using Test, Random, Enzyme
import KernelDispatch
using BFloat16s: BFloat16
import LuxTriangleAttention: Lux
const LTAg = LuxTriangleAttention

if GPU !== nothing
    # Perturbed LayerNorm affines and biases, so a dropped or misplaced term shows.
    perturb(ps) = Lux.fmap(a -> a isa Array{Float32} ? a .+ 0.3f0 .* randn(Random.Xoshiro(length(a)), Float32, size(a)) : a, ps)
    relerr_max(y, r) = maximum(abs.(Array(y) .- Array(r))) / maximum(abs.(Array(r)))

    function trimul_case(C, H, N, B; maskshape=:full, seed=1, kw...)
        plain = TriangleMultiplication(C, H; kw...)
        kern = TriangleMultiplication(C, H; use_kernel=true, kw...)
        ps, st = Lux.setup(Random.Xoshiro(seed), plain)
        ps = perturb(ps)
        rng = Random.Xoshiro(1000 + N + seed)
        x = randn(rng, Float32, C, N, N, B)
        mask = maskshape === :full ? rand(rng, N, N, B) .< 0.8 :
               maskshape === :shared ? rand(rng, N, N) .< 0.8 : nothing
        # The state too: `LayerNormNoBias` keeps its frozen bias there.
        dev(t) = Lux.fmap(a -> a isa Array ? DeviceArray(a) : a, t)
        maskd = mask === nothing ? nothing : DeviceArray(mask)
        return plain, kern, dev(ps), dev(st), DeviceArray(x), maskd
    end

    function check_case(C, H, N, B; tol=1.0e-5, kw...)
        plain, kern, ps, st, x, mask = trimul_case(C, H, N, B; kw...)
        r, _ = plain(x, mask, ps, st)
        y, st2 = kern(x, mask, ps, st)
        @test y isa DeviceArray
        @test size(y) == size(r)
        @test st2 == st
        return relerr_max(y, r) < tol
    end

    @testset "TriMul :default against the layer" begin
        @testset "direction, mask, batch, bias" begin
            for is_outgoing in (true, false), maskshape in (:full, :shared, :none), B in (1, 2), use_bias in (true, false)
                @test check_case(32, 24, 37, B; is_outgoing, maskshape, use_bias)
            end
        end
        @testset "parameter layouts" begin
            for fused in (true, false), fused_glu in (true, false), layer_norm_bias in (true, false)
                @test check_case(32, 16, 21, 2; fused, fused_glu, layer_norm_bias, use_bias=true)
            end
        end
        # T = 130²·1 = 16900 tokens: with the block pinned to nb = 4096 (a backend's tuned `nb` can exceed T), four full
        # blocks and a tail of 516.
        @testset "several blocks with a tail" begin
            LTAg.set_trimul_tunables!(; nb=4096)
            try
                @test LTAg._trimul_tunables(backend_tag()).nb < 130^2
                @test check_case(16, 8, 130, 1; is_outgoing=false)
            finally
                LTAg.set_trimul_tunables!()
            end
        end
        # C = 256: a [257, 32] Float32 tile and its reduction buffers take 33 KiB, more than Metal's 32 KiB of
        # threadgroup memory, so there the tile shrinks (CUDA's 48 KiB of static shared memory holds it).
        @testset "the LayerNorm tile fits local memory" begin
            p = LTAg._trimul_point(backend_tag(), 256, 32)
            @test 4 * (257 * p.TB + p.NT + 2p.TB) <= LTAg.KernelDispatch.device_caps(backend_tag()).max_shmem
            @test check_case(256, 32, 19, 1)
        end
    end

    # 16-bit data: the kernel against the layer's own forward in the same precision (their roundings differ in the order
    # of the 16-bit matrix products, hence the tolerance), and against Float32 on the rounded inputs. The contraction
    # modes: `HalfSafe` (unscaled parameters; the layer contracts in Float32), `Folded` (pre-scaled parameters), and a
    # problem whose contraction overflows Float16 unless it is done in Float32.
    @testset "16-bit data" begin
        cast(T, t) = Lux.fmap(a -> a isa Array{Float32} ? T.(a) : a, t)
        function half_case(T, C, H, N, B; fold=nothing, xscale=1.0f0, maskshape=:full, kw...)
            plain, kern, ps, st, x, mask = trimul_case(C, H, N, B; maskshape, kw...)
            ps, st = Lux.fmap(Array, ps), Lux.fmap(Array, st)
            x = xscale .* Array(x)
            if fold !== nothing
                ps, st = LTAg.fold_trimul(ps, st, fold)
            end
            dev(t) = Lux.fmap(a -> a isa Array ? DeviceArray(a) : a, t)
            ps16, st16, x16 = dev(cast(T, ps)), dev(st), DeviceArray(T.(x))
            r, _ = plain(x16, mask, ps16, st16)
            y, _ = kern(x16, mask, ps16, st16)
            # Float32 on the rounded values (the plain layer is only exact enough away from overflow)
            ref32, _ = plain(DeviceArray(Float32.(Array(x16))), mask, dev(cast(Float32, cast(T, ps))), st16)
            @test y isa DeviceArray && eltype(y) === T
            @test all(isfinite, Array(y))
            return relerr_max(y, r), relerr_max(y, ref32), relerr_max(r, ref32)
        end
        for T in (Float16, BFloat16)
            tol = T === Float16 ? 5.0f-3 : 3.0f-2      # measured: 1.6e-3 and 1.3e-2 at worst
            @testset "$T" begin
                @testset "HalfSafe, direction, mask, bias" begin
                    for is_outgoing in (true, false), maskshape in (:full, :none), use_bias in (true, false)
                        kr, k32, r32 = half_case(T, 32, 24, 21, 2; is_outgoing, maskshape, use_bias)
                        @test kr < tol
                        @test k32 < max(tol, 3 * r32)
                    end
                end
                @testset "parameter layouts" begin
                    for fused in (true, false), fused_glu in (true, false), layer_norm_bias in (true, false)
                        kr, = half_case(T, 32, 16, 17, 1; fused, fused_glu, layer_norm_bias, use_bias=true)
                        @test kr < tol
                    end
                end
                @testset "Folded" begin
                    kr, = half_case(T, 32, 24, 21, 2; fold=2, use_bias=true)
                    @test kr < tol
                end
                @testset "several blocks with a tail" begin
                    kr, = half_case(T, 16, 8, 130, 1; is_outgoing=false)
                    @test kr < tol
                end
            end
        end
        @testset "Float16 contraction that overflows" begin
            # |X| ~ 1e5 in Float16 unless the contraction runs in Float32 (src/precision.jl)
            kr, k32, r32 = half_case(Float16, 32, 24, 48, 1; xscale=60.0f0)
            @test kr < 5.0f-2
            @test k32 < 0.1
        end
    end

    # The backward, called directly (`KernelDispatch.backward`, the function the Enzyme rule calls; Enzyme cannot
    # differentiate the Metal code around it, so the rule itself is tested on CUDA, `test/ext/cuda/`). The reference is
    # Enzyme over the layer's own forward on the host, with the same cotangent.
    @testset "backward" begin
        refloss(plain, st, mask, w, x, ps) = sum(w .* first(plain(x, mask, ps, st)))
        leaves(t, p="") = t isa NamedTuple ? reduce(vcat, [leaves(t[k], p * "." * string(k)) for k in keys(t)]; init=Pair{String,Any}[]) :
                          t isa AbstractArray ? [p => t] : Pair{String,Any}[]
        relerr_(a, b) = maximum(abs.(Array(a) .- b)) / (maximum(abs.(b)) + 1.0f-12)

        function bwd_case(C, H, N, B; maskshape=:full, seed=1, tol=2.0f-4, kw...)
            plain = TriangleMultiplication(C, H; kw...)
            kern = TriangleMultiplication(C, H; use_kernel=true, kw...)
            ps, st = Lux.setup(Random.Xoshiro(seed), plain)
            ps = perturb(ps)
            rng = Random.Xoshiro(2000 + N + seed)
            x = randn(rng, Float32, C, N, N, B); w = randn(rng, Float32, C, N, N, B)
            mask = maskshape === :full ? rand(rng, N, N, B) .< 0.8 : maskshape === :shared ? rand(rng, N, N) .< 0.8 : nothing
            dx = zero(x); dps = Lux.fmap(zero, ps)
            Enzyme.autodiff(set_runtime_activity(Reverse), Const(refloss), Active, Const(plain), Const(st), Const(mask), Const(w),
                Duplicated(x, dx), Duplicated(ps, dps))
            dev(t) = Lux.fmap(a -> a isa Array ? DeviceArray(a) : a, t)
            args = (kern, DeviceArray(x), mask === nothing ? nothing : DeviceArray(mask), dev(ps), dev(st))
            y, st2 = kern(args[2:end]...)
            g = KernelDispatch.backward(TriMul(), LTAg.TriMulStaged(), args, (y, st2), (DeviceArray(w), nothing))
            @test g[1] === nothing && g[3] === nothing && g[5] === nothing
            ok = relerr_(g[2], dx) < tol
            ref, got = Dict(leaves(dps)), Dict(leaves(g[4]))
            @test keys(ref) == keys(got)
            for (k, v) in ref
                ok &= relerr_(got[k], v) < tol
            end
            return ok
        end

        @testset "direction, mask, batch, bias" begin
            for is_outgoing in (true, false), maskshape in (:full, :shared, :none), B in (1, 2), use_bias in (true, false)
                @test bwd_case(16, 12, 9, B; is_outgoing, maskshape, use_bias)
            end
        end
        @testset "parameter layouts" begin
            for fused in (true, false), fused_glu in (true, false), layer_norm_bias in (true, false)
                @test bwd_case(16, 8, 7, 2; fused, fused_glu, layer_norm_bias, use_bias=true)
            end
        end
        # 130² = 16900 tokens: two blocks of 16384, the second a tail of 516, so the weight gradients sum blocks.
        @testset "several blocks with a tail" begin
            @test bwd_case(16, 8, 130, 1; is_outgoing=false, use_bias=true, tol=1.0f-3)
        end
        @testset "the LayerNorm tile fits local memory" begin
            @test bwd_case(256, 32, 9, 1)
        end
        # The rule itself (KernelDispatch's Enzyme extension), in split mode as in `sdpa_op.gpu.test.jl`: Enzyme cannot
        # differentiate the Metal code around the op, so the cotangent is seeded by hand. It needs a KernelDispatch
        # whose rule accumulates into a `NamedTuple` of parameters (`_accumulate!`); an older one is skipped.
        @testset "through the Enzyme rule" begin
            ext = Base.get_extension(KernelDispatch, :KernelDispatchEnzymeExt)
            if ext === nothing || !isdefined(ext, :_accumulate!)
                @test_skip false
            else
                C, H, N, B = 8, 6, 5, 2
                plain = TriangleMultiplication(C, H; use_bias=true)
                kern = TriangleMultiplication(C, H; use_bias=true, use_kernel=true)
                ps, st = Lux.setup(Random.Xoshiro(1), plain)
                ps = perturb(ps)
                rng = Random.Xoshiro(2)
                x = randn(rng, Float32, C, N, N, B); w = randn(rng, Float32, C, N, N, B); mask = rand(rng, N, N, B) .< 0.8
                dx = zero(x); dps = Lux.fmap(zero, ps)
                Enzyme.autodiff(set_runtime_activity(Reverse), Const(refloss), Active, Const(plain), Const(st), Const(mask),
                    Const(w), Duplicated(x, dx), Duplicated(ps, dps))
                dev(t) = Lux.fmap(a -> a isa Array ? DeviceArray(a) : a, t)
                xd, md, psd, std = DeviceArray(x), DeviceArray(mask), dev(ps), dev(st)
                f(m, x, mask, ps, st) = first(m(x, mask, ps, st))
                gx, gps = DeviceArray(zeros(Float32, size(x))), dev(Lux.fmap(zero, ps))
                fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, Const{typeof(kern)},
                    Duplicated{typeof(xd)}, Const{typeof(md)}, Duplicated{typeof(psd)}, Const{typeof(std)})
                args = (Const(f), Const(kern), Duplicated(xd, gx), Const(md), Duplicated(psd, gps), Const(std))
                tape, _, shadow = fwd(args...)
                shadow .= DeviceArray(w)
                rev(args..., tape)
                @test relerr_(gx, dx) < 2.0f-4
                ref, got = Dict(leaves(dps)), Dict(leaves(gps))
                @test keys(ref) == keys(got)
                @test all(k -> relerr_(got[k], ref[k]) < 2.0f-4, keys(ref))
            end
        end
        @testset "deterministic" begin
            plain, kern, ps, st, x, mask = trimul_case(16, 8, 130, 1; use_bias=true)
            w = DeviceArray(randn(Random.Xoshiro(5), Float32, 16, 130, 130, 1))
            args = (kern, x, mask, ps, st)
            g1 = KernelDispatch.backward(TriMul(), LTAg.TriMulStaged(), args, nothing, (w, nothing))
            g2 = KernelDispatch.backward(TriMul(), LTAg.TriMulStaged(), args, nothing, (w, nothing))
            @test Array(g1[2]) == Array(g2[2])
            @test all(((a, b),) -> Array(a.second) == Array(b.second), zip(leaves(g1[4]), leaves(g2[4])))
        end
        @testset "declined calls throw, pointing to use_kernel=false" begin
            plain, kern, ps, st, x, mask = trimul_case(16, 8, 7, 1)
            ps16 = Lux.f16(ps)
            err = try
                KernelDispatch.backward(TriMul(), LTAg.TriMulStaged(), (kern, Float16.(x), mask, ps16, st), nothing,
                    (Float16.(x), nothing))
            catch e
                e
            end
            @test err isa ArgumentError && occursin("use_kernel=false", err.msg)
        end
    end

    @testset "TriMul declines, the layer's forward runs" begin
        plain, kern, ps, st, x, mask = trimul_case(32, 16, 13, 1)
        # Float16 data with Float32 parameters (mixed precision)
        x16 = Float16.(x)
        @test Array(first(kern(x16, mask, ps, st))) ≈ Array(first(plain(x16, mask, ps, st)))
        # Float64
        ps64, x64 = Lux.fmap(a -> a isa Array ? a : a, ps), x
        @test size(first(kern(x64, mask, ps64, st))) == size(x)
    end
end
