# trifast's Triton metallibs as a kernel of LuxTriangleAttention's `SDPA` op: the worked example of
# bringing your own attention kernel (`docs/custom-kernels.md`). A kernel type, its launch size, and
# `forward`/`backward` methods that hand each binary its arguments in the order it takes them. The
# launch size is split as KernelDispatch does it: each binary's `tile` (`register_trifast!`) and the call's
# `extent`.
# Used by `test/bench/kernels/registry_setup.jl`.

using LuxTriangleAttention
import KernelDispatch
using KernelDispatch: KernelBinary, OutArray, call_binary

"""
    Trifast

trifast's four Triton kernels: a forward, then a backward split three ways — `:dq` (which also
writes `delta`), `:dkv`, and a dedicated `:dbias`. They hard-code the outer extent to `n` (square
problems only) and are compiled with 32×32 tiles and unmasked stores (so `n` must be a multiple of
32). They read every operand at full size: an absent bias is a zero bias, an absent mask all-live.
"""
struct Trifast <: KernelDispatch.AbstractKernel end
KernelDispatch.kernelname(::Trifast) = :trifast

const TRIFAST_TILE = 32

# Every launch covers `extent = (n, n, H·B)`. A binary's `tile` is `(32, 1, 1)` (32 query rows per
# group), so it launches trifast's `(n/32, n, H·B)` grid; `:dbias` tiles both pair axes, `(32, 32, 1)`,
# so `(n/32, n/32, H·B)`.
_tri_extent(d) = (d.Nq, d.Nq, d.H * d.B)

# The types each metallib was compiled for, as `forward`/`backward` below bind them (`KernelBinary(...;
# signature=)`): trifast's metallibs are Float32, a Bool mask, and `(scale, neg_inf, n, h)` after the buffers.
const TRIFAST_SIGNATURES = let F = Vector{Float32}, M = Vector{Bool}, sc = (Float32, Float32, Int32, Int32)
    (fwd=(F, F, F, F, F, F, M, sc...),                      # o, lse, q, k, v, bias, mask
        dq=(F, F, F, F, F, F, M, F, F, F, sc...),           # delta, dq, q, k, v, bias, mask, dO, o, lse
        dkv=(F, F, F, F, F, F, M, F, F, F, sc...),          # dk, dv, q, k, v, bias, mask, dO, lse, delta
        dbias=(F, F, F, F, F, M, F, F, F, sc...))           # dbias, q, k, v, bias, mask, dO, lse, delta
end

# Every launch takes `(scale, neg_inf, n, h)` after its buffers.
_tri_scalars(d) = (Float32(inv(sqrt(d.hd))), -floatmax(Float32), Int32(d.Nq), Int32(d.H))
function _tri_prep(args...)
    r = LuxTriangleAttention._sdpa_prep(args..., false)
    d = r[6]
    d.Nq == d.Nj || throw(ArgumentError("trifast runs square problems only (Nq == Nj)"))
    d.Nq % TRIFAST_TILE == 0 || throw(ArgumentError(
        "trifast tiles n in blocks of $TRIFAST_TILE and has unmasked stores; n = $(d.Nq) is not a multiple"))
    return r
end

function KernelDispatch.forward(op::LuxTriangleAttention.SDPA, k::Trifast, q0, k0, v0, mask, bias)
    LTA = LuxTriangleAttention
    q, kk, v, m, b, d = _tri_prep(q0, k0, v0, mask, bias)
    o, lse = call_binary(op, k, :fwd, OutArray(d.T, LTA._full(d)), OutArray(Float32, LTA._res(d)),
        q, kk, v, b, m, _tri_scalars(d)...; key=(nameof(d.T), d.hd), extent=_tri_extent(d))
    return LTA._finish(o, lse, q0)
end

function KernelDispatch.backward(op::LuxTriangleAttention.SDPA, k::Trifast, (q0, k0, v0, mask, bias),
        (o, lse), (dO, _))
    LTA = LuxTriangleAttention
    q, kk, v, m, b, d = _tri_prep(q0, k0, v0, mask, bias)
    o, dO, lse = reshape(o, LTA._full(d)), reshape(dO, LTA._full(d)), reshape(lse, LTA._res(d))
    sc, key, extent = _tri_scalars(d), (nameof(d.T), d.hd), _tri_extent(d)
    delta, dq = call_binary(op, k, :dq, OutArray(Float32, LTA._res(d)), OutArray(d.T, LTA._full(d)),
        q, kk, v, b, m, dO, o, lse, sc...; key, extent)
    dk, dv = call_binary(op, k, :dkv, OutArray(d.T, LTA._full(d)), OutArray(d.T, LTA._full(d)),
        q, kk, v, b, m, dO, lse, delta, sc...; key, extent)
    db, = call_binary(op, k, :dbias, OutArray(Float32, (d.Nq, d.Nq, d.H, d.B); zero=true),
        q, kk, v, b, m, dO, lse, delta, sc...; key, extent)
    return (LTA._as(dq, q0), LTA._as(dk, k0), LTA._as(dv, v0), nothing,
        bias === nothing ? nothing : LTA._as(db, bias))
end

