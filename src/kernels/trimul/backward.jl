# The backward of `TriMul`'s `:default` kernel (`TriMulStaged`): the reverse of the three stages in `op.jl`, in the same
# block-wise, workspace-owning style. docs/FEATURE_trimul_kernel.md has the design and the measurements.
#
# It keeps nothing from the forward but the inputs: `xn`, `y`, `V`, `G` and the projection `P` are recomputed block by
# block (the paper's choice, arXiv 2610.05854 App. J), and `a`, `b`, `X` are recomputed whole, because the contraction is
# global. The weight gradients are block-summed GEMMs (`mul!` with β = 1) in a fixed block order, and the LayerNorms'
# scale and bias gradients are per-group partial sums (written by the backward LayerNorm kernel) reduced by a GEMV with
# ones, so the result is deterministic (no atomics).
#
#   A  recompute      a, b (K1, as the forward), X = a ⊗ b
#   B  per block      xn = LN_C(x); y = LN_H(X); V = W_o·y; G = W_g·xn
#                     dV = dout·σ(G), dG = dout·V·σ'(G)                       :gate_bwd, in place on V, G
#                     dW_o += dV·yᵀ, dW_g += dG·xnᵀ, db_o += Σ dV, db_g += Σ dG   GEMMs / GEMVs
#                     dy = W_oᵀ·dV;  dxn = W_gᵀ·dG  -> the block of dx (the gate path of the input's cotangent)
#                     dX = LN_H⁻¹(dy), written over X;  dγ_o, dβ_o                :ln_out_bwd
#   C  contraction    dA, dB from dX, a, b (batched GEMMs, the transposes of the forward's)
#   D  per block      xn, P recomputed; dP from dA, dB, P                          :glu_bwd, in place on P
#                     dW_ab += dPᵀ·xnᵀ, db_ab += Σ dP;  t = W_abᵀ·dPᵀ
#                     dx = LN_C⁻¹(dx + t), in place;  dγ_i, dβ_i                  :ln_in_bwd (adds `t` itself)
#
# Every GEMM and GEMV runs on buffers at offset zero (Metal.jl takes a slow path for a view that starts mid-buffer: 12 ms
# against 0.4 ms for a 128 × 16384 GEMV), so the blocks of `dx` are filled by `copyto!` and read by the kernels.
#
# The memory peak is the inputs plus a, b, X (3 units of N²·H), the cotangent dx (1) and, for the contraction's
# backward, dA and dB (2): `a`, `b` and `X` go as soon as dA and dB exist.

# The tunables of the backward stages: the LayerNorm backward holds TWO `[D + 1, TB]` tiles (the input and the
# cotangent) and two reduction buffers, so its tile shrinks further than the forward's.
function _trimul_bwd_point(be, C, H)
    p = _trimul_tunables(be)
    D, shmem = max(C, H), device_caps(be).max_shmem
    TB = p.TB
    while TB > 1 && 4 * (2 * (D + 1) * TB + 2 * p.NT + 4TB) > shmem
        TB ÷= 2
    end
    return merge(p, (; TB))
end

# Variant key of the backward's binaries (`:bwd` marks it; its Float32 is the only type it covers).
_trimul_bwd_key(C, H, p, hb, hm) = (:bwd, :Float32, C, H, p.TB, p.NT, p.NTE, hb, hm)

function _build_bwd!(k::TriMulStaged, key)
    _, _, C, H, TB, NT, NTE, hb, hm = key
    p = (; TB, NT, NTE)
    be = KernelDispatch.device_backend()
    bins = map(TRIMUL_BWD_STAGES) do stage
        name = replace("lta_trimul_$(stage)_$(KernelDispatch.source_tag(k))", r"[^A-Za-z0-9_]" => "_")
        stage => KernelDispatch.ka_compile(be, trimul_stage_kernel(stage), trimul_argtypes(stage, C, H, p, hb, hm);
            tg=trimul_stage_threads(stage, p), name, tile=(trimul_stage_tile(stage, p), 1, 1),
            params=(; TB, NT, NTE), dir=_staging_dir())
    end
    KernelDispatch.add_variant!(TriMul(), k, key; bins...)
    return nothing
end

"""
    trimul_args(Val(stage), a, s) -> Tuple

For the backward stages `:gate_bwd`, `:glu_bwd`, `:ln_out_bwd` and `:ln_in_bwd`, the roles are `V`, `G`, `dout`, `bo`, `bg`
(gate), `P`, `da`, `db`, `bias`, `mask` (GLU), `dx`, `xin`, `dy`, `dy2`, `γ`, `part` (LayerNorm; `dx` an `InOutArray`, `dy2` added to `dy` by `:ln_in_bwd` only) and the scalars
`t0`, `n`, `T`, `mmod`, `ldx`, `dyoff`, `ε`.
"""
trimul_args(::Val{:gate_bwd}, a, s) = (a.V, a.G, a.dout, a.bo, a.bg, s.t0, s.n)
trimul_args(::Val{:glu_bwd}, a, s) = (a.P, a.da, a.db, a.bias, a.mask, s.t0, s.n, s.T, s.mmod)
trimul_args(::Val{:ln_out_bwd}, a, s) = (a.dx, a.xin, a.dy, a.dy2, a.γ, a.part, s.t0, s.n, s.ldx, s.dyoff, s.ε)
trimul_args(::Val{:ln_in_bwd}, a, s) = (a.dx, a.xin, a.dy, a.dy2, a.γ, a.part, s.t0, s.n, s.ldx, s.dyoff, s.ε)

# Why the backward cannot run this call, or `nothing`: the forward's reasons, and the precision (Float32, plain contraction).
function _trimul_bwd_unsupported(m, x, mask, ps, st, dout)
    why = _trimul_unsupported(m, x, mask, ps, st)
    isnothing(why) || return why
    eltype(x) === Float32 || return "the backward runs Float32 data only (got $(eltype(x)))"
    contraction_mode(x, st.core) isa FullPrecision || return "the backward does not cover folded (`fold_trimul`) parameters"
    size(dout) == size(x) && eltype(dout) === Float32 || return "the output cotangent must be Float32 and shaped like the input"
    return nothing
end

# The flat range of columns `t0+1 … t0+n` of a `[C, T]` buffer. It is only ever copied or read by a kernel: a GEMM or GEMV
# on a buffer at a non-zero offset takes Metal.jl's slow path (12 ms against 0.4 ms for a 128 × 16384 GEMV), so every
# product runs on offset-zero blocks (`_block`, `view(buf, :, 1:n)`).
_colrange(C, t0, n) = (t0 * C + 1):((t0 + n) * C)

# dA, dB of the contraction `X = a ⊗ b`, the transposes of `_contract`'s products (planes `[N, N, B·H]`).
#   outgoing  X = A·Bᵀ per plane:  dA = dX·B,   dB = dXᵀ·A
#   incoming  X = Aᵀ·B per plane:  dA = B·dXᵀ,  dB = A·dX
_contract_bwd_a(::True, dX, a, b) = Lux.batched_matmul(dX, b)
_contract_bwd_a(::False, dX, a, b) = Lux.batched_matmul(b, Lux.batched_transpose(dX))
_contract_bwd_b(::True, dX, a, b) = Lux.batched_matmul(Lux.batched_transpose(dX), a)
_contract_bwd_b(::False, dX, a, b) = Lux.batched_matmul(a, dX)

"""
    KernelDispatch.backward(::TriMul, ::TriMulStaged, (m, x, mask, ps, st), outs, (dout, _))
        -> (nothing, dx, nothing, dps, nothing)

The reverse of the staged forward: the cotangent `dx` of the input and `dps`, the gradient of every parameter, shaped
like `ps`. Float32 only, for every parameter layout the forward covers; other calls throw, pointing to
`use_kernel=false` (the layer's own forward, which any AD differentiates).
"""
function KernelDispatch.backward(op::TriMul, k::TriMulStaged, (m, x, mask, ps, st), _, (dout, _))
    why = _trimul_bwd_unsupported(m, x, mask, ps, st, dout)
    isnothing(why) || throw(ArgumentError(
        "`TriMul`'s `:default` backward does not run this call ($why): build the `TriangleMultiplication` with " *
        "`use_kernel=false` to differentiate it"))
    KernelDispatch.tracing() && throw(ArgumentError(
        "`TriMul`'s `:default` kernel runs eagerly only for now: build the layer with `use_kernel=false` to trace it"))

    core = m.core
    C, N, _, B = size(x)
    H, T = core.chn_hidden, N * N * B
    be = backend_of(x)
    p, pb = _trimul_point(be, C, H), _trimul_bwd_point(be, C, H)
    w = _trimul_params(m, ps, st)
    hm = mask !== nothing
    kf, kb = _trimul_key(Float32, Float32, C, H, p, w.hb, hm), _trimul_bwd_key(C, H, pb, w.hb, hm)
    nb = min(p.nb, T)
    ng = cld(nb, pb.TB)                                              # LayerNorm groups of a block

    ws(dims...) = similar(x, Float32, dims...)
    zs(dims...) = fill!(ws(dims...), 0.0f0)
    ph = ws(1)
    one_vec = fill!(ws(max(nb, ng)), 1.0f0)
    ones_(n) = view(one_vec, 1:n)
    x2, do2 = reshape(x, C, T), reshape(dout, C, T)
    io = KernelDispatch.InOutArray
    launchf(stage, buf, s; extent) = call_binary(op, k, stage, trimul_args(Val(stage), buf, s)...; key=kf, extent)
    launchb(stage, buf, s; extent) = call_binary(op, k, stage, trimul_args(Val(stage), buf, s)...; key=kb, extent)
    mmask = mask === nothing ? similar(x, Bool, 1) : reshape(mask, :)
    s0 = (; T=Int32(T), C=Int32(C), mmod=Int32(mask === nothing ? 1 : length(mask)), εi=w.εi, εo=w.εo)
    block(t0) = (n = min(nb, T - t0); (n, merge(s0, (; t0=Int32(t0), n=Int32(n)))))
    mm! = LinearAlgebra.mul!

    # A: recompute a, b and X.
    xn, P = ws(C, nb), ws(nb * 4H)
    a, b = ws(T, H), ws(T, H)
    abuf = (; xn=io(xn), x=x2, w.γi, w.βi, a=io(a), b=io(b), bias=something(w.bab, ph), mask=mmask)
    _trimul_k1!(launchf, w, abuf, s0, P, xn, nb, T, H)
    pa, pbb = reshape(a, N, N, B * H), reshape(b, N, N, B * H)
    X = _contract(core.is_outgoing, pa, pbb)
    Xt = reshape(X, T, H)

    # The gradients (accumulated by the blocks) and the cotangent of the input.
    dxv = ws(C * T)
    dWab, dbab = zs(size(w.Wab)...), w.bab === nothing ? nothing : zs(4H)
    dWo, dbo = zs(size(w.Wo)...), w.bo === nothing ? nothing : zs(C)
    dWg, dbg = zs(size(w.Wg)...), w.bg === nothing ? nothing : zs(C)
    gi, go = zs(2C), zs(2H)                                          # (dγ, dβ) of the input / output LayerNorm

    # B: the output stage, per block.
    y, V, G, dyb = ws(H, nb), ws(C, nb), ws(C, nb), ws(H, nb)
    part, dxb = ws(2 * max(C, H) * ng), ws(C, nb)
    obuf = (; xn=io(xn), x=x2, w.γi, w.βi, y=io(y), X=Xt, w.γo, w.βo, V=io(V), G=io(G), dout=do2,
        bo=something(w.bo, ph), bg=something(w.bg, ph))
    for t0 in 0:nb:(T - 1)
        n, s = block(t0)
        Vn, Gn, yn, xnn, dyn, dxn = view(V, :, 1:n), view(G, :, 1:n), view(y, :, 1:n), view(xn, :, 1:n),
            view(dyb, :, 1:n), view(dxb, :, 1:n)
        launchf(:ln_in, obuf, s; extent=n)
        launchf(:ln_out, obuf, s; extent=n)
        mm!(Vn, w.Wo, yn)
        mm!(Gn, w.Wg, xnn)
        launchb(:gate_bwd, obuf, s; extent=n * C)                      # V, G now hold dV, dG
        mm!(dWo, Vn, transpose(yn), 1.0f0, 1.0f0)
        mm!(dWg, Gn, transpose(xnn), 1.0f0, 1.0f0)
        dbo === nothing || mm!(dbo, Vn, ones_(n), 1.0f0, 1.0f0)
        dbg === nothing || mm!(dbg, Gn, ones_(n), 1.0f0, 1.0f0)
        mm!(dyn, transpose(w.Wo), Vn)
        mm!(dxn, transpose(w.Wg), Gn)
        copyto!(view(dxv, _colrange(C, t0, n)), view(vec(dxb), 1:(C * n)))      # the gate path's share of dx
        lb = (; dx=io(Xt), xin=Xt, dy=dyb, dy2=ph, γ=w.γo, part)
        launchb(:ln_out_bwd, lb, merge(s, (; ldx=Int32(T), dyoff=Int32(0), ε=w.εo)); extent=n)   # dX over X
        ngn = cld(n, pb.TB)
        mm!(go, _block(part, 2H, ngn), ones_(ngn), 1.0f0, 1.0f0)
    end
    foreach(z -> release!(be, z), (y, V, G, dyb))

    # C: the contraction's backward. a and b go once dA and dB exist.
    dX = reshape(Xt, N, N, B * H)
    da3 = _contract_bwd_a(core.is_outgoing, dX, pa, pbb)
    db3 = _contract_bwd_b(core.is_outgoing, dX, pa, pbb)
    foreach(z -> release!(be, z), (pa, pbb, a, b, dX, Xt, X))
    dat, dbt = reshape(da3, T, H), reshape(db3, T, H)

    # D: the input stage, per block.
    dbuf = (; xn=io(xn), x=x2, w.γi, w.βi, da=dat, db=dbt, bias=something(w.bab, ph), mask=mmask)
    for t0 in 0:nb:(T - 1)
        n, s = block(t0)
        xnn, Pb, dxn = view(xn, :, 1:n), _block(P, n, 4H), view(dxb, :, 1:n)
        launchf(:ln_in, dbuf, s; extent=n)
        mm!(Pb, transpose(xnn), transpose(w.Wab))
        launchb(:glu_bwd, merge(dbuf, (; P=Pb)), s; extent=n * H)       # Pb now holds dP
        dbab === nothing || mm!(dbab, transpose(Pb), ones_(n), 1.0f0, 1.0f0)
        mm!(dWab, transpose(Pb), transpose(xnn), 1.0f0, 1.0f0)
        mm!(dxn, transpose(w.Wab), transpose(Pb))                        # the projection's share of dxn
        lb = (; dx=io(dxv), xin=x2, dy=dxv, dy2=dxb, γ=w.γi, part)        # dxn = the gate path's (in dx) + this
        launchb(:ln_in_bwd, lb, merge(s, (; ldx=Int32(C), dyoff=Int32(t0), ε=w.εi)); extent=n)
        ngn = cld(n, pb.TB)
        mm!(gi, _block(part, 2C, ngn), ones_(ngn), 1.0f0, 1.0f0)
    end
    foreach(z -> release!(be, z), (xn, P, part, dxb, one_vec, da3, db3, dat, dbt))

    dps = _trimul_grads(m, ps, (; dγi=gi[1:C], dβi=gi[(C + 1):(2C)], dγo=go[1:H], dβo=go[(H + 1):(2H)],
        dWab, dbab, dWo, dbo, dWg, dbg))
    return (nothing, reshape(dxv, size(x)), nothing, dps, nothing)
end

# --- the parameter gradients, in the layer's own layout -------------------------------------------------------------------
#
# The inverse of `_trimul_params`: the gradient arrays above assembled into a `NamedTuple` shaped like `ps` (zeros for
# whatever the kernel does not use, such as `LayerNormNoBias`'s frozen bias).

function _trimul_grads(m, ps, g)
    core, pc = m.core, ps.core
    z = Lux.fmap(zero, ps)
    glu_out = (; linear=_dense_grad(pc.glu_out.linear, g.dWo, g.dbo), gate=_dense_grad(pc.glu_out.gate, g.dWg, g.dbg))
    dcore = merge(z.core, _ab_grads(core, pc, g.dWab, g.dbab),
        (; layer_norm_out=_ln_grad(pc.layer_norm_out, g.dγo, g.dβo), glu_out=merge(z.core.glu_out, glu_out)))
    return merge(z, (; layer_norm=_ln_grad(ps.layer_norm, g.dγi, g.dβi), core=dcore))
end

_ln_grad(p, dγ, dβ) = merge(Lux.fmap(zero, p), (; scale=reshape(dγ, size(p.scale))),
    haskey(p, :bias) ? (; bias=reshape(dβ, size(p.bias))) : (;))
_dense_grad(p, dW, db) = haskey(p, :bias) ? (; weight=dW, bias=db) : (; weight=dW)

# The fused `a`/`b` projection's rows are (a values, b values, a gates, b gates).
_ab_grads(core::TriMulCore{True}, pc, dW, db) = (; glu_ab=_fused_ab_grads(core.glu_ab.layer, pc.glu_ab, dW, db))
_fused_ab_grads(::GatedLinearUnit{True}, p, dW, db) = (; linear=_dense_grad(p.linear, dW, db))
function _fused_ab_grads(::GatedLinearUnit{False}, p, dW, db)       # one GLU, a separate gate Dense
    n = size(dW, 1) ÷ 2
    return (; linear=_dense_grad(p.linear, dW[1:n, :], _rows(db, 1:n)),
        gate=_dense_grad(p.gate, dW[(n + 1):(2n), :], _rows(db, (n + 1):(2n))))
end
function _ab_grads(core::TriMulCore{False}, pc, dW, db)             # separate GLUs for a and b
    H = core.chn_hidden
    r(i) = ((i - 1) * H + 1):(i * H)
    part(i) = (dW[r(i), :], _rows(db, r(i)))
    return (; glu_a=_glu_grads(core.glu_a.layer, pc.glu_a, part(1), part(3)),
        glu_b=_glu_grads(core.glu_b.layer, pc.glu_b, part(2), part(4)))
end
_rows(::Nothing, r) = nothing
_rows(v, r) = v[r]
_glu_grads(::GatedLinearUnit{True}, p, (Wv, bv), (Wg, bg)) =
    (; linear=_dense_grad(p.linear, vcat(Wv, Wg), bv === nothing ? nothing : vcat(bv, bg)))
_glu_grads(::GatedLinearUnit{False}, p, (Wv, bv), (Wg, bg)) =
    (; linear=_dense_grad(p.linear, Wv, bv), gate=_dense_grad(p.gate, Wg, bg))
