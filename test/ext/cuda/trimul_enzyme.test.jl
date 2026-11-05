# `TriangleMultiplication(use_kernel=true)` differentiated END TO END by Enzyme on CUDA: the reverse rule of KernelDispatch's
# Enzyme extension (which accumulates the layer's parameter gradients field by field) calling the staged `TriMul` backward, with
# Enzyme differentiating the CUDA.jl code around the op (the loss). On Metal this is not possible (Enzyme cannot differentiate
# Metal.jl code), so there the rule is tested in split mode in `test/kernels/trimul_op.gpu.test.jl`.
#
# Needs a KernelDispatch whose Enzyme extension has `_accumulate!`; an older one is skipped.

using Test, Random, Enzyme
import KernelDispatch
import LuxTriangleAttention: Lux
using LuxTriangleAttention: TriangleMultiplication

trimul_hostloss(plain, st, mask, w, x, ps) = sum(w .* first(plain(x, mask, ps, st)))
trimul_devloss(kern, st, mask, w, x, ps) = sum(w .* first(kern(x, mask, ps, st)))

@testset "TriMul through Enzyme (CUDA)" begin
    ext = Base.get_extension(KernelDispatch, :KernelDispatchEnzymeExt)
    if ext === nothing || !isdefined(ext, :_accumulate!)
        @test_skip false
    else
        leaves(t, p="") = t isa NamedTuple ? reduce(vcat, [leaves(t[k], p * "." * string(k)) for k in keys(t)]; init=Pair{String,Any}[]) :
                          t isa AbstractArray ? [p => t] : Pair{String,Any}[]
        relerr_(a, b) = maximum(abs.(Array(a) .- b)) / (maximum(abs.(b)) + 1.0f-12)
        for is_outgoing in (true, false), (C, H, N, B) in ((16, 12, 9, 2), (16, 8, 130, 1))
            plain = TriangleMultiplication(C, H; is_outgoing, use_bias=true)
            kern = TriangleMultiplication(C, H; is_outgoing, use_bias=true, use_kernel=true)
            ps, st = Lux.setup(Random.Xoshiro(1), plain)
            ps = Lux.fmap(a -> a isa Array{Float32} ? a .+ 0.3f0 .* randn(Random.Xoshiro(length(a)), Float32, size(a)) : a, ps)
            rng = Random.Xoshiro(2)
            x = randn(rng, Float32, C, N, N, B); w = randn(rng, Float32, C, N, N, B); mask = rand(rng, N, N, B) .< 0.8

            dx, dps = zero(x), Lux.fmap(zero, ps)
            Enzyme.autodiff(set_runtime_activity(Reverse), Const(trimul_hostloss), Active, Const(plain), Const(st),
                Const(mask), Const(w), Duplicated(x, dx), Duplicated(ps, dps))

            dev(t) = Lux.fmap(a -> a isa Array ? DeviceArray(a) : a, t)
            xd, psd = DeviceArray(x), dev(ps)
            gx, gps = DeviceArray(zero(x)), dev(Lux.fmap(zero, ps))
            Enzyme.autodiff(set_runtime_activity(Reverse), Const(trimul_devloss), Active, Const(kern), Const(dev(st)),
                Const(DeviceArray(mask)), Const(DeviceArray(w)), Duplicated(xd, gx), Duplicated(psd, gps))

            @testset "outgoing=$is_outgoing N=$N" begin
                @test relerr_(gx, dx) < 1.0f-3
                ref, got = Dict(leaves(dps)), Dict(leaves(gps))
                @test keys(ref) == keys(got)
                @test all(k -> relerr_(got[k], ref[k]) < 1.0f-3, keys(ref))
            end
        end
    end
end
