# Triangle multiplication as a KernelDispatch op: `TriMul`, and its `:default` kernel `TriMulStaged`.
#
# The kernel follows TurboPairFormer's three-stage design (arXiv 2610.05854, App. J, Fig. 12) with the matrix
# products left to the vendor (MPS, cuBLAS) and everything around them in this package's stage kernels
# (`trimul/kernels.jl`). It works on blocks of `nb` tokens, so the `[T, 4H]` projection never exists whole, and it
# owns its workspaces and frees them as it goes: eagerly, a forward's peak memory is the sum of what it allocates
# (the GC does not run inside a call), so that is what decides the peak. docs/FEATURE_trimul_kernel.md has the design
# and the measurements.
#
#   K1, per block:  xn = LN_C(x)                    :ln_in
#                   P  = xnᵀ·W_abᵀ                   GEMM          [n, 4H]
#                   a, b = mask·P_v·σ(P_g)           :glu          -> a, b [T, H] token-major
#   K2:             X = a ⊗ b per channel            `_contract`   (batched GEMM, as the layer)
#   K3, per block:  y  = LN_H(X)                     :ln_out
#                   V  = W_o·y;  G = W_g·LN_C(x)     GEMMs         (the gate's LayerNorm is recomputed: :ln_in)
#                   out = (V + b_o)·σ(G + b_g)       :gate         -> out [C, T] channel-first
#
# 16-bit data (Float16, BFloat16) runs the same stages with Float32 arithmetic inside the kernels; the contraction
# mode (`src/precision.jl`) only decides the type of a, b and X: `HalfSafe` keeps them in Float32, `Folded` and Float32
# data in the data's type. What the op does not cover runs the layer's own forward (`_trimul_fallback`): other
# activations, mixed precision, and a Reactant trace (the traced layer is the faster program there). The backward (`trimul/backward.jl`) covers Float32.

"""
    TriMul

The triangle-multiplication op: `TriMul()(layer, x, mask, ps, st) -> (y, st)`, the forward of a
[`TriangleMultiplication`](@ref) `layer`. Host arrays run the layer's own forward; device arrays the selected
kernel (`KernelDispatch.use_kernel!(TriMul, name)`), `:default` unless changed. A layer routes here when built
with `use_kernel=true`.

The op takes the whole layer, not its arrays: the kernel fuses the input LayerNorm, reads every parameter of the
layer, and runs the layer's own forward for a configuration it does not cover.
"""
struct TriMul <: KernelDispatch.AbstractKernelOp end
KernelDispatch.opname(::TriMul) = :trimul
KernelDispatch.host(::TriMul, m, x, mask, ps, st) = _trimul_layer(m, x, mask, ps, st)

"""
    TriMulStaged

`:default`, `TriMul`'s kernel: three stages (input, contraction, output) with vendor GEMMs for the matrix products
and this package's KernelAbstractions kernels for the LayerNorms, gates and mask, over blocks of tokens. Portable:
the same code runs on every GPU backend; a backend chooses only its tile and block sizes
([`trimul_tunables`](@ref)).

Covers `Float32`, `Float16` and `BFloat16` data (parameters of the same type) with sigmoid gates, for every
parameter layout of the layer (`fused`, `fused_glu`, `use_bias`, `layer_norm_bias`), every [`ContractionMode`](@ref)
(plain, [`HalfSafe`](@ref), [`Folded`](@ref)) and a `[N, N, B]`, `[N, N]` or absent mask. All arithmetic is `Float32`,
the matrix products run in the storage type. Other configurations run the layer's own forward, with a note logged
once; so does a Reactant trace (the traced layer is the faster program). The backward
([`KernelDispatch.backward`](@ref), `trimul/backward.jl`) covers `Float32` only.
"""
struct TriMulStaged <: KernelDispatch.AbstractKernel end
KernelDispatch.kernelname(::TriMulStaged) = :default
KernelDispatch.@default_kernel TriMul TriMulStaged()

# A manifest written from other kernel sources is dropped: the binaries are rebuilt rather than served stale.
const TRIMUL_SOURCE_TAG = string(hash(read(joinpath(@__DIR__, "kernels.jl"), String)) % UInt32; base=36)
const _TRIMUL_KEY_FORMAT = "t2"
KernelDispatch.source_tag(::TriMulStaged) = TRIMUL_SOURCE_TAG * "-" * _TRIMUL_KEY_FORMAT

# --- what the kernel covers --------------------------------------------------------------------------------------------

# Why the kernel cannot run this call, or `nothing`.
function _trimul_unsupported(m, x, mask, ps, st)
    ndims(x) == 4 || return "it takes a rank-4 `[C, N, N, B]` input"
    T = eltype(x)
    T in (Float32, Float16, BFloat16) || return "it runs Float32, Float16 and BFloat16 data (got $T)"
    _all_eltype(ps, T) || return "its parameters must have the data's type ($T)"
    core = m.core
    _ln_activation(m.layer_norm) === identity && _ln_activation(core.layer_norm_out) === identity ||
        return "its LayerNorms must have no activation"
    all(g -> g.activation === Lux.sigmoid, _glus(core)) || return "its gates must be sigmoids"
    C, N, _, B = size(x)
    Int64(N)^2 * B * max(C, 4 * core.chn_hidden) < typemax(Int32) || return "N²·B·max(C, 4H) must fit Int32"
    if mask !== nothing
        eltype(mask) === Bool || return "its mask must be Bool"
        length(mask) in (N^2, N^2 * B) || return "its mask must be [N, N, B] or [N, N]"
    end
    return nothing
end

_all_eltype(x::AbstractArray, ::Type{T}) where {T} = eltype(x) === T
_all_eltype(x::Union{NamedTuple,Tuple}, ::Type{T}) where {T} = all(v -> _all_eltype(v, T), values(x))
_all_eltype(_, ::Type) = true
_ln_activation(l) = _inner_layernorm(l).activation
_glus(core::TriMulCore{True}) = (core.glu_ab.layer, core.glu_out.layer)
_glus(core::TriMulCore{False}) = (core.glu_a.layer, core.glu_b.layer, core.glu_out.layer)

function _trimul_fallback(why, m, x, mask, ps, st)
    @info "`TriMul`'s `:default` kernel does not run this call ($why); running the layer's own forward" maxlog = 1
    return _trimul_layer(m, x, mask, ps, st)
end

# --- the parameters, as the kernel reads them --------------------------------------------------------------------------

# A LayerNorm's scale and bias as vectors. `LayerNormNoBias` keeps its (zero) bias frozen in the state.
function _ln_affine(p, s)
    γ = vec(p.scale)
    haskey(p, :bias) && return γ, vec(p.bias)
    fp = s isa NamedTuple ? get(s, :frozen_params, (;)) : (;)
    # The state need not have been cast with the parameters (`Lux.f16(ps)` leaves it): the kernel wants one type.
    return γ, haskey(fp, :bias) ? eltype(γ).(vec(fp.bias)) : fill!(similar(γ), 0)
end

# The `a`/`b` projection as ONE `[4H, C]` weight with rows (a values, b values, a gates, b gates), and its bias (or
# `nothing`). The default layout (`fused = fused_glu = true`) IS that weight, so it is used as is; the others are
# assembled, a copy of `4H·C` numbers per call.
_ab_weight(core::TriMulCore{True}, pc) = _fused_ab_weight(core.glu_ab.layer, pc.glu_ab)
_fused_ab_weight(::GatedLinearUnit{True}, p) = p.linear.weight, get(p.linear, :bias, nothing)
function _fused_ab_weight(::GatedLinearUnit{False}, p)                  # one GLU, separate gate Dense
    v, g = p.linear, p.gate
    return vcat(v.weight, g.weight), haskey(v, :bias) ? vcat(v.bias, g.bias) : nothing
end
function _ab_weight(core::TriMulCore{False}, pc)                          # separate GLUs for a and b
    av, ag, ab = _glu_parts(core.glu_a.layer, pc.glu_a)
    bv, bg, bb = _glu_parts(core.glu_b.layer, pc.glu_b)
    W = vcat(av.weight, bv.weight, ag.weight, bg.weight)
    return W, ab ? vcat(av.bias, bv.bias, ag.bias, bg.bias) : nothing
end
# One GLU's value and gate Dense parameters, and whether they have biases.
function _glu_parts(g::GatedLinearUnit{True}, p)
    n, l = g.out_chn, p.linear
    hb = haskey(l, :bias)
    v = (; weight=l.weight[1:n, :], (hb ? (; bias=l.bias[1:n]) : (;))...)
    gt = (; weight=l.weight[(n + 1):(2n), :], (hb ? (; bias=l.bias[(n + 1):(2n)]) : (;))...)
    return v, gt, hb
end
_glu_parts(::GatedLinearUnit{False}, p) = p.linear, p.gate, haskey(p.linear, :bias)

function _trimul_params(m, ps, st, mode=FullPrecision())
    core, pc, sc = m.core, ps.core, st.core
    γi, βi = _ln_affine(ps.layer_norm, st.layer_norm)
    γo, βo = _ln_affine(pc.layer_norm_out, sc.layer_norm_out)
    Wab, bab = _ab_weight(core, pc)
    lo, lg = pc.glu_out.linear, pc.glu_out.gate
    hb = bab !== nothing
    return (; γi, βi, εi=Float32(_inner_layernorm(m.layer_norm).epsilon),
        γo, βo, εo=_eps_out(mode, Float32(_inner_layernorm(core.layer_norm_out).epsilon)),
        Wab, bab, Wo=lo.weight, bo=get(lo, :bias, nothing), Wg=lg.weight, bg=get(lg, :bias, nothing), hb)
end

# The output LayerNorm's ε. A `Folded(k)` layer has X scaled by 4⁻ᵏ, so its variance is 16⁻ᵏ of the unscaled one and ε
# scales with it (`_normalize_out(::Folded)` in `tri_mul_core.jl`): the LayerNorm then returns exactly the unscaled result.
_eps_out(::ContractionMode, ε) = ε
_eps_out(mode::Folded, ε) = ε * 16.0f0^(-mode.k)

# --- tunables, variants, building -------------------------------------------------------------------------------------

# The tunables for this backend, with the LayerNorm tile shrunk until its local memory fits the device:
# `[D + 1, TB]` + `[NT]` + `[2, TB]` Float32 for the larger of C and H.
function _trimul_point(be, C, H)
    p = _trimul_tunables(be)
    D, shmem = max(C, H), device_caps(be).max_shmem
    TB = p.TB
    while TB > 1 && 4 * ((D + 1) * TB + p.NT + 2TB) > shmem
        TB ÷= 2
    end
    return merge(p, (; TB))
end

# Variant key: storage type of the data, of a/b/X, channels, hidden channels, the compiled tunables and whether a bias /
# mask is present (absent operands are separate binaries that never read them).
_trimul_key(T, Ta, C, H, p, hb, hm) = (nameof(T), nameof(Ta), C, H, p.TB, p.NT, p.NTE, hb, hm)
_storage(name::Symbol) = name === :BFloat16 ? BFloat16 : getfield(Base, name)

# A variant is compiled on first use (no tuning: the block size, a host-side choice, is the one that matters; see
# the feature doc). Four binaries, one per stage, built for the loaded device backend.
function KernelDispatch.build!(::TriMul, k::TriMulStaged, key)
    key[1] === :bwd && return _build_bwd!(k, key)
    Tn, Tan, C, H, TB, NT, NTE, hb, hm = key
    T, Ta = _storage(Tn), _storage(Tan)
    p = (; TB, NT, NTE)
    be = KernelDispatch.device_backend()
    bins = map(TRIMUL_STAGES) do stage
        name = replace("lta_trimul_$(stage)_$(KernelDispatch.source_tag(k))", r"[^A-Za-z0-9_]" => "_")
        stage => KernelDispatch.ka_compile(be, trimul_stage_kernel(stage), trimul_argtypes(stage, C, H, p, hb, hm, T, Ta);
            tg=trimul_stage_threads(stage, p), name, tile=(trimul_stage_tile(stage, p), 1, 1),
            params=(; TB, NT, NTE), dir=_staging_dir())
    end
    KernelDispatch.add_variant!(TriMul(), k, key; bins...)
    return nothing
end

"""
    trimul_args(Val(stage), a, s) -> Tuple

The arguments of one `TriMul` stage binary, in slot order — the one place they are written. `a` holds the buffers
by role (`xn`, `x`, `γi`, `βi`, `y`, `X`, `γo`, `βo`, `a`, `b`, `P`, `bias`, `mask`, `out`, `V`, `G`, `bo`, `bg`;
an output is an `InOutArray`), `s` the scalars (`t0`, `n`, `T`, `C`, `mmod`, `εi`, `εo`). The order is
[`trimul_argtypes`](@ref)'s.
"""
trimul_args(::Val{:ln_in}, a, s) = (a.xn, a.x, a.γi, a.βi, s.t0, s.n, s.C, s.εi)
trimul_args(::Val{:ln_out}, a, s) = (a.y, a.X, a.γo, a.βo, s.t0, s.n, s.T, s.εo)
trimul_args(::Val{:glu}, a, s) = (a.a, a.b, a.P, a.bias, a.mask, s.t0, s.n, s.T, s.mmod)
trimul_args(::Val{:gate}, a, s) = (a.out, a.V, a.G, a.bo, a.bg, s.t0, s.n)

# --- the forward ------------------------------------------------------------------------------------------------------

function KernelDispatch.forward(op::TriMul, k::TriMulStaged, m, x, mask, ps, st)
    # Under a Reactant trace the layer's own forward is traced: XLA fuses its elementwise work and schedules its products
    # itself, and the traced layer is faster than this eager pipeline (Metal, AlphaFold3 sizes, N = 512: 88 ms against
    # 114 ms). The kernel's block loops also write their workspaces in place (`InOutArray`), which jax-mps cannot trace.
    if KernelDispatch.tracing()
        @info "`TriMul`'s `:default` kernel is an eager pipeline; under a Reactant trace the layer's own forward is traced" maxlog = 1
        return _trimul_layer(m, x, mask, ps, st)
    end
    why = _trimul_unsupported(m, x, mask, ps, st)
    isnothing(why) || return _trimul_fallback(why, m, x, mask, ps, st)

    core = m.core
    C, N, _, B = size(x)
    H, T = core.chn_hidden, N * N * B
    be = backend_of(x)
    Tx = eltype(x)
    mode = contraction_mode(x, st.core)
    Ta = trimul_contraction_type(be, Tx, mode)                      # a, b, X: what the contraction runs in
    p = _trimul_point(be, C, H)
    w = _trimul_params(m, ps, st, mode)
    key = _trimul_key(Tx, Ta, C, H, p, w.hb, mask !== nothing)
    nb = min(p.nb, T)

    ws(dims...) = similar(x, Tx, dims...)                            # workspaces on x's device
    wa(dims...) = similar(x, Ta, dims...)
    ph = ws(1)                                                       # an absent bias's placeholder
    x2 = reshape(x, C, T)
    out = ws(C, N, N, B)
    xn, P = ws(C, nb), ws(nb * 4H)
    a, b = wa(T, H), wa(T, H)
    io = KernelDispatch.InOutArray
    launch(stage, buf, s; extent) = call_binary(op, k, stage, trimul_args(Val(stage), buf, s)...; key, extent)
    s0 = (; T=Int32(T), C=Int32(C), mmod=Int32(mask === nothing ? 1 : length(mask)), εi=w.εi, εo=w.εo)
    abuf = (; xn=io(xn), x=x2, w.γi, w.βi, a=io(a), b=io(b), bias=something(w.bab, ph),
        mask=mask === nothing ? similar(x, Bool, 1) : reshape(mask, :))

    # K1: LayerNorm, the a/b projection, gates and mask, block by block.
    _trimul_k1!(launch, w, abuf, s0, P, xn, nb, T, H)
    release!(be, P)

    # K2: the contraction, as the layer runs it. A reshape of a device array shares its buffer's reference count, so
    # each one is released too, or the buffer stays until the GC runs.
    pa, pb = reshape(a, N, N, B * H), reshape(b, N, N, B * H)
    X = _contract(core.is_outgoing, pa, pb)
    foreach(z -> release!(be, z), (pa, pb, a, b))
    Xt = reshape(X, T, H)

    # K3: LayerNorm over H, the output projection, the recomputed gate, the gated store.
    y, V, G = ws(H, nb), ws(C, nb), ws(C, nb)
    obuf = (; xn=io(xn), x=x2, w.γi, w.βi, y=io(y), X=Xt, w.γo, w.βo, out=io(reshape(out, C, T)), V, G,
        bo=something(w.bo, ph), bg=something(w.bg, ph))
    for t0 in 0:nb:(T - 1)
        n = min(nb, T - t0)
        s = merge(s0, (; t0=Int32(t0), n=Int32(n)))
        launch(:ln_out, obuf, s; extent=n)
        LinearAlgebra.mul!(view(V, :, 1:n), w.Wo, view(y, :, 1:n))
        launch(:ln_in, obuf, s; extent=n)
        LinearAlgebra.mul!(view(G, :, 1:n), w.Wg, view(xn, :, 1:n))
        launch(:gate, obuf, s; extent=n * C)
    end
    foreach(z -> release!(be, z), (xn, y, V, G, Xt, X))
    return out, st
end

# K1 of the forward, and of the backward's recomputation: `a`, `b` (in `abuf`) from the input, block by block.
function _trimul_k1!(launch, w, abuf, s0, P, xn, nb, T, H)
    for t0 in 0:nb:(T - 1)
        n = min(nb, T - t0)
        s = merge(s0, (; t0=Int32(t0), n=Int32(n)))
        Pb = _block(P, n, 4H)
        launch(:ln_in, abuf, s; extent=n)
        LinearAlgebra.mul!(Pb, transpose(view(xn, :, 1:n)), transpose(w.Wab))
        launch(:glu, merge(abuf, (; P=Pb)), s; extent=n * H)
    end
    return nothing
end

# A contiguous `[n, m]` block from the front of a flat buffer. `view(P, 1:n, :)` would be strided, and Metal.jl
# then skips MPS for the product (10× slower); cuBLAS would take the leading dimension, so this costs nothing there.
_block(buf, n, m) = reshape(view(buf, 1:(n * m)), n, m)
