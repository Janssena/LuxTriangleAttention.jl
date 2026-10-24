# `Attention(...; use_kernel=true)` on Metal arrays, once per fused kernel, against the same layer and parameters with
# `use_kernel=false` on the same arrays. Julia's own array code on the device is trusted; what is ours is the kernel the
# layer hands its q/k/v to, so that is the only difference between the two runs. The traced version of the same
# comparison, and eager against traced, is `test/ext/reactant/attention_kernel.test.jl`.
#
# The kernel accumulates in Float32 and the plain path (LuxLib) in the storage type, so the bound at Float16 is the
# plain path's own rounding. The two must not be bitwise equal: equal outputs would mean the kernel did not run.

using Test, Random
import LuxTriangleAttention: Lux, Static.static

relL2_k(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))

# Switching the kernel takes effect from the next top-level expression; `invokelatest` runs `f` after the switch.
function with_attention_kernel(f, kernel, n, hd, ::Type{T}) where {T}
    prev = LuxTriangleAttention.current_kernel()
    try
        use_kernel!(kernel)
        kernel === :default ? tune_flash_kernel!(n; dim=hd, T) : tune_flash_mma!(n; dim=hd, T)
        return Base.invokelatest(f)
    finally
        use_kernel!(prev)
    end
end

@testset "Attention on Metal, fused kernel against no kernel" begin
    C, hd, H, B = 32, 8, 4, 2
    N = 13
    cases = (
        ("sequence [C,N,B]", (C, N, B), :kq, (N, B)),
        ("MSA row [C,N,S,B]", (C, N, 7, B), :kq, (N, 7, B)),
        ("pair [C,N,N,B]", (C, N, N, B), :qk, (N, N, B)),
    )
    dm(a) = a isa AbstractArray ? DeviceArray(a) : a isa NamedTuple ? map(dm, a) : a

    @testset "$name, $T, fuse_qkv=$fuse_qkv, :$kernel" for (name, xdims, layout, mdims) in cases,
            T in (Float16, Float32), fuse_qkv in (true, false), kernel in (:default, :flash_mma)

        rng = Random.Xoshiro(21)
        x = randn(rng, T, xdims...)
        bias = randn(rng, T, H, N, N, B)
        mask = rand(rng, Bool, mdims...)
        plain = Attention(C, hd, H; bias_layout=layout, fuse_qkv)
        fused = Attention(C, hd, H; bias_layout=layout, fuse_qkv, use_kernel=static(true))
        ps, st = Lux.setup(rng, plain) |> convert_types(T)
        args = (dm(x), dm(bias), dm(mask), dm(ps), st)

        y_plain = Array(first(plain(args...)))
        run() = Array(first(fused(args...)))
        # A plain sequence's fused q/k/v reach the kernel as a `reshape` of a `view` of device memory, a wrapper
        # pair KernelDispatch's Metal extension does not recognise (it knows each alone): the launch is refused.
        if length(xdims) == 3 && fuse_qkv
            @test_broken isapprox(with_attention_kernel(run, kernel, N, hd, T), y_plain)
        else
            y = with_attention_kernel(run, kernel, N, hd, T)
            @test eltype(y) === T
            @test all(isfinite, y)
            @test relL2_k(y, y_plain) < (T === Float16 ? 2.0e-2 : 1.0e-4)
            @test y != y_plain
        end
    end
end
