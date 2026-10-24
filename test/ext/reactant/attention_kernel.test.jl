# The ordinary `Attention` layer on the fused kernel (`use_kernel=true`), against the SAME layer and
# parameters on the plain softmax path (`use_kernel=false`), both traced on the jax-mps Metal client, and the traced
# kernel against the same kernel launched eagerly on device arrays (`attention.gpu.test.jl` is the eager kernel against
# no kernel).
#
# Triangle attention is the one shape the kernel was written for (`Nq == Nj`) and is covered by
# `kernel_launch.test.jl`. This file is everything else the layer is used for:
#
#   * a plain sequence `[C, N, B]` — rank-4 q, reshaped to an outer extent of 1 (`_seq5`);
#   * MSA row attention `[C, N, S, B]` with a `:kq` pair bias `[H, N, N, B]` — attends over `N` for
#     each of `S` rows, `S` both above and below `N` so neither extent can stand in for the other;
#   * column-style attention with NO bias — the launcher substitutes a zero bias.
#
# Each case also asserts the kernel's custom_call is in the IR: the host path is numerically
# identical, so a quiet fallback would pass every comparison while testing nothing.

using Test, Random, Reactant, Enzyme
import LuxTriangleAttention: Lux, Static.static
# Guarded: re-including would reset `mps_client`'s cache, and a SECOND client for the same device
# fails to load — which returned `nothing` and silently skipped this whole file.

const _MPS_ATT = traced_client()

# One sparse hole per outer slice, never a whole key row: a fully-masked row softmaxes to NaN on
# both paths, which would compare equal and hide everything else.
function _holes(dims...)
    m = fill(true, dims...)
    m[2, ntuple(Returns(1), length(dims) - 1)...] = false
    m[end, ntuple(i -> dims[i+1], length(dims) - 1)...] = false
    return m
end

if _MPS_ATT === nothing
    @warn "jax-mps Metal PJRT plugin not found; skipping Attention-on-kernel tests" maxlog = 1
elseif GPU === nothing
    @warn "Metal not functional; skipping Attention-on-kernel tests" maxlog = 1
else
    const _PREV_BACKEND_ATT = Reactant.XLA.default_backend()
    Reactant.set_default_backend(_MPS_ATT)
    try
        @testset "Attention(use_kernel=true) == use_kernel=false" begin
            T = Float32
            C, B = 24, 1
            cases = [
                # (name, head_dim, heads, x dims, bias?, mask?)
                ("sequence [C,N,B], bias + mask", 8, 4, (C, 13, B), true, true),
                ("sequence [C,N,B], no bias", 16, 2, (C, 13, B), false, true),
                ("MSA row [C,N,S,B], S > N, bias + mask", 8, 4, (C, 7, 11, B), true, true),
                ("MSA row [C,N,S,B], S < N, bias + mask", 32, 2, (C, 12, 5, B), true, true),
                ("column-style [C,S,N,B], no bias, mask", 8, 4, (C, 9, 6, B), false, true),
                ("MSA row, no mask", 16, 2, (C, 10, 3, B), true, false),
            ]
            for (name, hd, H, xdims, with_bias, with_mask) in cases
                @testset "$name" begin
                    rng = Random.Xoshiro(11)
                    N = xdims[2]                     # the attended extent
                    LuxTriangleAttention.tune_flash_kernel!(N; dim=hd, T)

                    lk = Attention(C, hd, H; use_kernel=static(true))
                    lh = Attention(C, hd, H; use_kernel=static(false))
                    ps, st = Lux.setup(rng, lh)

                    x = randn(rng, T, xdims...)
                    bias = with_bias ? randn(rng, T, H, N, N, B) : nothing
                    mask = with_mask ? _holes(xdims[2:end]...) : nothing

                    dev = Reactant.to_rarray
                    xd, psd, std = dev(x), dev(ps), dev(st)
                    bd = bias === nothing ? nothing : dev(bias)
                    md = mask === nothing ? nothing : dev(mask)
                    f(l, x, b, m, p, s) = first(l(x, b, m, p, s))

                    yk = Array(@jit f(lk, xd, bd, md, psd, std))
                    yh = Array(@jit f(lh, xd, bd, md, psd, std))
                    y_host = first(lh(x, bias, mask, ps, st))   # plain Julia, CPU

                    @test size(yk) == size(x)
                    @test yk ≈ yh rtol = 1.0f-5 atol = 1.0f-6
                    @test yk ≈ y_host rtol = 1.0f-5 atol = 1.0f-6

                    # …and the kernel embedded in the trace is the one an eager launch on device arrays runs. (A plain
                    # sequence `[C, N, B]` cannot be launched eagerly with its fused q/k/v: they arrive as a `reshape`
                    # of a `view` of device memory, which KernelDispatch's Metal extension does not recognise.)
                    dm(a) = a isa AbstractArray ? DeviceArray(a) : a isa NamedTuple ? map(dm, a) : a
                    eager() = Array(first(lk(dm(x), dm(bias), dm(mask), dm(ps), st)))
                    if ndims(x) == 3
                        @test_broken isapprox(eager(), yk; rtol=1.0f-5, atol=1.0f-6)
                    else
                        @test eager() ≈ yk rtol = 1.0f-5 atol = 1.0f-6
                    end

                    ir = repr(Reactant.@code_hlo optimize = false f(lk, xd, bd, md, psd, std))
                    @test occursin("stablehlo.custom_call", ir)
                end
            end
        end

        @testset "16-bit storage: result eltypes and values (traced)" begin
            # The traced path once declared EVERY custom-call result `Float32`. The declared type is
            # the buffer jax-mps allocates, so at `Float16` the kernel wrote halves into a buffer read
            # back as floats: the wrong type AND garbage values, with no error. The result types now
            # follow `_role_eltype` (`o` in storage type, `lse` Float32), as on the eager path. See
            # docs/done/BUG_traced_kernel_16bit_results.md.
            relL2(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))
            hd, H, N, S, B = 16, 4, 12, 7, 1
            dev = Reactant.to_rarray
            fo(q, k, v, m, b) = first(LuxTriangleAttention.SDPA()(q, k, v, m, b))
            fl(q, k, v, m, b) = last(LuxTriangleAttention.SDPA()(q, k, v, m, b))
            for T in (Float32, Float16, LuxTriangleAttention.BFloat16), with_bias in (true, false),
                with_mask in (true, false)
                @testset "$(nameof(T)), bias=$with_bias, mask=$with_mask" begin
                    rng = Random.Xoshiro(5)
                    LuxTriangleAttention.tune_flash_kernel!(N; dim=hd, T)
                    T === Float32 || LuxTriangleAttention.tune_flash_kernel!(N; dim=hd, T=Float32)
                    r(dims...) = T.(randn(rng, Float32, dims...))      # `randn` has no BFloat16 method
                    q, k, v = (r(hd, H, N, S, B) for _ in 1:3)
                    bias = with_bias ? r(N, N, H, 1, B) : nothing
                    mask = with_mask ? _holes(N, 1, 1, S, B) : nothing
                    tol = T === Float32 ? 1e-6 : T === Float16 ? 1e-3 : 1e-2
                    up(x) = x === nothing ? nothing : Float32.(x)
                    dd(x) = x === nothing ? nothing : dev(x)

                    o = @jit fo(dev(q), dev(k), dev(v), dd(mask), dd(bias))
                    lse = @jit fl(dev(q), dev(k), dev(v), dd(mask), dd(bias))
                    @test eltype(o) === T
                    @test eltype(lse) === Float32
                    # The Float32 kernel on the same values, upcast: with Float32 accumulation the
                    # only difference is input/output rounding.
                    ref = Array(@jit fo(dev(up(q)), dev(up(k)), dev(up(v)), dd(mask), dd(up(bias))))
                    ref_lse = Array(@jit fl(dev(up(q)), dev(up(k)), dev(up(v)), dd(mask), dd(up(bias))))
                    @test all(isfinite, Array(o))
                    @test relL2(Array(o), ref) < tol
                    @test relL2(Array(lse), ref_lse) < tol
                    @test occursin("stablehlo.custom_call",
                        repr(Reactant.@code_hlo optimize = false fo(dev(q), dev(k), dev(v), dd(mask), dd(bias))))
                end
            end

            @testset "Float32 bias beside Float16 activations is converted" begin
                # The kernel binds the bias as a pointer to the storage type; an unconverted Float32
                # bias would be read as reinterpreted bytes.
                rng = Random.Xoshiro(6)
                LuxTriangleAttention.tune_flash_kernel!(N; dim=hd, T=Float16)
                q, k, v = (randn(rng, Float16, hd, H, N, S, B) for _ in 1:3)
                bias32 = randn(rng, Float32, N, N, H, 1, B)
                mask = _holes(N, 1, 1, S, B)
                got = @jit fo(dev(q), dev(k), dev(v), dev(mask), dev(bias32))
                want = Array(@jit fo(dev(q), dev(k), dev(v), dev(mask), dev(Float16.(bias32))))
                @test eltype(got) === Float16
                @test Array(got) == want
            end

            @testset "Attention layer keeps Float16" begin
                rng = Random.Xoshiro(7)
                C, ahd, AH = 24, 8, 4
                LuxTriangleAttention.tune_flash_kernel!(N; dim=ahd, T=Float16)
                lk = Attention(C, ahd, AH; use_kernel=static(true))
                lh = Attention(C, ahd, AH; use_kernel=static(false))
                ps, st = Lux.setup(rng, lh)
                ps16 = Lux.fmap(x -> x isa AbstractArray{Float32} ? Float16.(x) : x, ps)
                x = randn(rng, Float16, C, N, S, B)
                bias = randn(rng, Float16, AH, N, N, B)
                mask = _holes(N, S, B)
                f(l, x, b, m, p, s) = first(l(x, b, m, p, s))
                yk = @jit f(lk, dev(x), dev(bias), dev(mask), dev(ps16), dev(st))
                yh = @jit f(lh, dev(x), dev(bias), dev(mask), dev(ps16), dev(st))
                @test eltype(yk) === Float16
                @test eltype(yh) === Float16
                @test relL2(Array(yk), Array(yh)) < 1e-2
            end
        end

        @testset "traced gradient through the kernel is a hard error" begin
            # Enzyme cannot differentiate the kernel's custom call under Reactant yet. Rather than
            # quietly run the materialising default attention in the differentiated program, the
            # traced call refuses (`traced_autodiff`); the kernel-free build still differentiates,
            # and a forward-only program keeps the custom call.
            T = Float32
            C, hd, H, N, S, B = 24, 8, 4, 9, 5, 1
            LuxTriangleAttention.tune_flash_kernel!(N; dim=hd, T)
            rng = Random.Xoshiro(3)
            lk = Attention(C, hd, H; use_kernel=static(true))
            lh = Attention(C, hd, H; use_kernel=static(false))
            ps, st = Lux.setup(rng, lh)
            x, bias, mask = randn(rng, T, C, N, S, B), randn(rng, T, H, N, N, B), _holes(N, S, B)
            dev = Reactant.to_rarray
            xd, bd, md, psd, std = dev(x), dev(bias), dev(mask), dev(ps), dev(st)
            loss(l, x, b, m, p, s) = sum(abs2, first(l(x, b, m, p, s)))
            g(l, x, b, m, p, s) = Enzyme.gradient(Enzyme.Reverse, loss, Enzyme.Const(l), x, Enzyme.Const(b),
                                                  Enzyme.Const(m), p, Enzyme.Const(s))
            err = try
                Reactant.@compile g(lk, xd, bd, md, psd, std)
                nothing
            catch e
                e
            end
            @test err !== nothing
            @test occursin("cannot be differentiated under Reactant", sprint(showerror, err))
            @test !LuxTriangleAttention.KERNEL_IN_TRACED_AUTODIFF[]
            gh = @jit g(lh, xd, bd, md, psd, std)                # the kernel-free build is fine
            @test all(isfinite, Array(gh[2]))
            f(l, x, b, m, p, s) = first(l(x, b, m, p, s))
            @test occursin("stablehlo.custom_call", repr(Reactant.@code_hlo optimize = false f(lk, xd, bd, md, psd, std)))
        end
    finally
        Reactant.set_default_backend(_PREV_BACKEND_ATT)
    end
end
