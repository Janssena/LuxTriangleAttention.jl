# The stage kernels of `TriMul`'s `:default` kernel (`TriMulStaged`, `trimul/op.jl`): everything of a
# `TriangleMultiplication` forward that is not a matrix product. The products themselves are vendor GEMMs
# (`mul!`, `_contract`) run by the host code between these launches.
#
# Backend-agnostic, like `flash/forward.jl`, and written under the same conventions (see its header):
#   * buffers are RAW POINTERS, read with `unsafe_load` and written with `unsafe_store!`, so the binary's ABI is the
#     plain one a custom call binds;
#   * the grid is 1-D and the kernel decodes its own indices;
#   * no code that can throw: runtime conversions are `_i32`, runtime division `_sdiv` / `_srem` (CUDA binaries
#     are built without the state an exception needs, and are refused otherwise);
#   * index arithmetic is Int32 (64-bit `÷` and `%` are emulated on GPUs and were half the elementwise time on
#     Metal); only the final linear offsets are Int64, so a large `C·N²·B` cannot overflow.
#
# Layouts (T = N²·B tokens, token t = i + N·j + N²·b, 0-based):
#   x, out      [C, T]  channel-first (the layer's own layout)
#   a, b, X     [T, H]  token-major: column h is the B contiguous [N, N] planes `_contract` takes
#   blocks      a block of n ≤ nb tokens from t0 on: xn / y channel-first [D, n], P token-major [n, 4H],
#               V / G channel-first [C, n]
#
# Storage is Float32, Float16 or BFloat16 (the element type of each buffer); ALL arithmetic is Float32: loads widen,
# stores narrow (`eltype(y)(v)`), and every reduction and statistic is Float32 whatever the storage.

"""
    trimul_ln!(y, x, γ, β, t0, n, ldx, eps, Val(D), Val(TB), Val(NT), Val(TOKMAJ))

LayerNorm over `D` channels of the `n` tokens `t0 + 1 … t0 + n` of `x`, written channel-first to the block
`y[:, 1:n]`. The source is channel-first `x[d, t]` (`TOKMAJ = false`, the input `x`) or token-major `x[t, d]`
with leading dimension `ldx` (`true`, the contraction result `X`). A group of `NT` threads handles `TB` tokens:
it stages the `[D, TB]` tile in local memory (padded by one row against bank conflicts), so both the read and
the write are coalesced whichever layout the source has, then reduces each token's mean and variance with
`NT ÷ TB` threads per token. Two-pass statistics, like `Lux.LayerNorm`.

No private value lives across a `@synchronize` (KernelAbstractions' semantics: its CPU backend splits the
kernel there); each segment re-derives the thread's indices.
"""
@kernel unsafe_indices = true function trimul_ln!(y, x, γ, β, t0::Int32, n::Int32, ldx::Int32, eps::Float32,
    ::Val{D}, ::Val{TB}, ::Val{NT}, ::Val{TOKMAJ}) where {D,TB,NT,TOKMAJ}
    tile = @localmem Float32 (D + 1, TB)
    part = @localmem Float32 (NT ÷ TB, TB)
    stat = @localmem Float32 (2, TB)

    tid = _i32(@index(Local, Linear)) - Int32(1)
    tb0 = (_i32(@index(Group, Linear)) - Int32(1)) * Int32(TB)
    e = tid
    while e < Int32(D * TB)
        if TOKMAJ                                  # consecutive threads on tokens: coalesced along t
            tt = _srem(e, Int32(TB)); d = _sdiv(e, Int32(TB))
        else                                       # consecutive threads on channels
            d = _srem(e, Int32(D)); tt = _sdiv(e, Int32(D))
        end
        t = tb0 + tt
        v = 0.0f0
        if t < n
            g = Int64(t0 + t)
            v = Float32(TOKMAJ ? unsafe_load(x, g + Int64(ldx) * d + 1) : unsafe_load(x, Int64(D) * g + d + 1))
        end
        @inbounds tile[d + 1, tt + 1] = v
        e += Int32(NT)
    end
    @synchronize

    tid = _i32(@index(Local, Linear)) - Int32(1)
    tok = _srem(tid, Int32(TB)); p = _sdiv(tid, Int32(TB))       # token `tok`, channels p, p + NT÷TB, …
    s = 0.0f0
    d = p
    while d < Int32(D)
        s += @inbounds tile[d + 1, tok + 1]
        d += Int32(NT ÷ TB)
    end
    @inbounds part[p + 1, tok + 1] = s
    @synchronize

    tid = _i32(@index(Local, Linear)) - Int32(1)
    if tid < Int32(TB)
        s = 0.0f0
        for q in 1:(NT ÷ TB)
            s += @inbounds part[q, tid + 1]
        end
        @inbounds stat[1, tid + 1] = s / Float32(D)
    end
    @synchronize

    tid = _i32(@index(Local, Linear)) - Int32(1)
    tok = _srem(tid, Int32(TB)); p = _sdiv(tid, Int32(TB))
    μ = @inbounds stat[1, tok + 1]
    s = 0.0f0
    d = p
    while d < Int32(D)
        δ = @inbounds(tile[d + 1, tok + 1]) - μ
        s += δ * δ
        d += Int32(NT ÷ TB)
    end
    @inbounds part[p + 1, tok + 1] = s
    @synchronize

    tid = _i32(@index(Local, Linear)) - Int32(1)
    if tid < Int32(TB)
        s = 0.0f0
        for q in 1:(NT ÷ TB)
            s += @inbounds part[q, tid + 1]
        end
        @inbounds stat[2, tid + 1] = inv(sqrt(s / Float32(D) + eps))
    end
    @synchronize

    tid = _i32(@index(Local, Linear)) - Int32(1)
    tb0 = (_i32(@index(Group, Linear)) - Int32(1)) * Int32(TB)
    e = tid
    while e < Int32(D * TB)
        d = _srem(e, Int32(D)); tt = _sdiv(e, Int32(D)); t = tb0 + tt
        if t < n
            v = @inbounds (tile[d + 1, tt + 1] - stat[1, tt + 1]) * stat[2, tt + 1]
            unsafe_store!(y, eltype(y)(v * Float32(unsafe_load(γ, d + 1)) + Float32(unsafe_load(β, d + 1))), Int64(D) * t + d + 1)
        end
        e += Int32(NT)
    end
end

@inline _sigmoid(x::Float32) = inv(1.0f0 + exp(-x))

"""
    trimul_glu!(a, b, P, bias, mask, t0, n, lda, mmod, Val(H), Val(NTE), Val(HB), Val(HM))

The gated, masked `a` and `b` of a block, from its projection `P` (`[n, 4H]`, columns `a` values, `b` values, `a`
gates, `b` gates — the rows of the fused `glu_ab` weight): `a[t, h] = keep·(P[t, h] + b₁)·σ(P[t, 2H+h] + b₃)` and
the same for `b` with columns `H+h` and `3H+h`, written token-major into `a`/`b` (leading dimension `lda`) at
token `t0 + t`. `keep = mask[(t0 + t) % mmod]`: a `[N, N, B]` mask has `mmod = N²·B`, a shared `[N, N]` one
`mmod = N²`. `HB`: add the bias; `HM`: read the mask. One thread per (token, channel), consecutive threads on
tokens, so every read and write is coalesced.
"""
@kernel unsafe_indices = true function trimul_glu!(a, b, P, bias, mask, t0::Int32, n::Int32, lda::Int32,
    mmod::Int32, ::Val{H}, ::Val{NTE}, ::Val{HB}, ::Val{HM}) where {H,NTE,HB,HM}
    i = (_i32(@index(Group, Linear)) - Int32(1)) * Int32(NTE) + _i32(@index(Local, Linear)) - Int32(1)
    if i < n * Int32(H)
        tt = _srem(i, n); h = _sdiv(i, n); t = t0 + tt
        keep = HM ? unsafe_load(mask, _srem(t, mmod) + 1) : true
        pn = Int64(n)
        p0 = Int64(tt) + 1
        av = Float32(unsafe_load(P, p0 + pn * h))
        bv = Float32(unsafe_load(P, p0 + pn * (Int32(H) + h)))
        ag = Float32(unsafe_load(P, p0 + pn * (Int32(2H) + h)))
        bg = Float32(unsafe_load(P, p0 + pn * (Int32(3H) + h)))
        if HB
            av += Float32(unsafe_load(bias, h + 1))
            bv += Float32(unsafe_load(bias, Int32(H) + h + 1))
            ag += Float32(unsafe_load(bias, Int32(2H) + h + 1))
            bg += Float32(unsafe_load(bias, Int32(3H) + h + 1))
        end
        o = Int64(t) + Int64(lda) * h + 1
        unsafe_store!(a, eltype(a)(keep ? av * _sigmoid(ag) : 0.0f0), o)
        unsafe_store!(b, eltype(b)(keep ? bv * _sigmoid(bg) : 0.0f0), o)
    end
end

"""
    trimul_gate!(out, V, G, bo, bg, t0, n, Val(C), Val(NTE), Val(HB))

The gated output of a block: `out[c, t0 + t] = (V[c, t] + b_o[c])·σ(G[c, t] + b_g[c])`, all channel-first, one
thread per element, consecutive threads on channels. `V` is the output projection of the normalised contraction,
`G` the gate projection of the normalised input; `HB` adds their biases.
"""
@kernel unsafe_indices = true function trimul_gate!(out, V, G, bo, bg, t0::Int32, n::Int32,
    ::Val{C}, ::Val{NTE}, ::Val{HB}) where {C,NTE,HB}
    i = (_i32(@index(Group, Linear)) - Int32(1)) * Int32(NTE) + _i32(@index(Local, Linear)) - Int32(1)
    if i < n * Int32(C)
        c = _srem(i, Int32(C)); tt = _sdiv(i, Int32(C))
        v = Float32(unsafe_load(V, Int64(i) + 1))
        g = Float32(unsafe_load(G, Int64(i) + 1))
        if HB
            v += Float32(unsafe_load(bo, c + 1))
            g += Float32(unsafe_load(bg, c + 1))
        end
        unsafe_store!(out, eltype(out)(v * _sigmoid(g)), Int64(C) * (t0 + tt) + c + 1)
    end
end

# --- the backward stages ------------------------------------------------------------------------------------------------
#
# The backward recomputes the forward's `xn`, `y`, `V`, `G`, `P` block by block (it keeps only `a`, `b` and `X`), so it
# reuses the forward stage kernels above for that, and these four for the derivatives. The matrix products of the
# backward (weight and input gradients) are vendor GEMMs again, between the launches (`trimul/backward.jl`).

"""
    trimul_gate_bwd!(V, G, dout, bo, bg, t0, n, Val(C), Val(NTE), Val(HB))

The output gate's derivatives of a block, IN PLACE: with `V` the output projection and `G` the gate projection
(`[C, n]`, channel-first; biases `bo`, `bg` added when `HB`) and `dout = dout[:, t0+1:t0+n]` the incoming cotangent
(`[C, T]`), `out = v·σ(g)` gives `dV = dout·σ(g)` and `dG = dout·v·σ(g)(1 − σ(g))`. They replace `V` and `G`.
"""
@kernel unsafe_indices = true function trimul_gate_bwd!(V, G, dout, bo, bg, t0::Int32, n::Int32,
    ::Val{C}, ::Val{NTE}, ::Val{HB}) where {C,NTE,HB}
    i = (_i32(@index(Group, Linear)) - Int32(1)) * Int32(NTE) + _i32(@index(Local, Linear)) - Int32(1)
    if i < n * Int32(C)
        c = _srem(i, Int32(C)); tt = _sdiv(i, Int32(C))
        v = Float32(unsafe_load(V, Int64(i) + 1))
        g = Float32(unsafe_load(G, Int64(i) + 1))
        if HB
            v += Float32(unsafe_load(bo, c + 1))
            g += Float32(unsafe_load(bg, c + 1))
        end
        d = Float32(unsafe_load(dout, Int64(C) * (t0 + tt) + c + 1))
        s = _sigmoid(g)
        unsafe_store!(V, eltype(V)(d * s), Int64(i) + 1)
        unsafe_store!(G, eltype(G)(d * v * s * (1.0f0 - s)), Int64(i) + 1)
    end
end

"""
    trimul_glu_bwd!(P, da, db, bias, mask, t0, n, lda, mmod, Val(H), Val(NTE), Val(HB), Val(HM))

The `a`/`b` gates' derivatives of a block, IN PLACE on the projection `P` (`[n, 4H]`, as [`trimul_glu!`](@ref) reads it):
`a = keep·u·σ(g)` gives `du = keep·da·σ(g)` and `dg = keep·da·u·σ(g)(1 − σ(g))`, the same for `b`, where `u`, `g` are the
projection's value and gate columns (plus their biases when `HB`). `da`, `db` are the cotangents of `a`, `b`
(`[T, H]` token-major, leading dimension `lda`, at token `t0 + t`). The result replaces `P`: columns
(`du_a`, `du_b`, `dg_a`, `dg_b`), the cotangent of the projection.
"""
@kernel unsafe_indices = true function trimul_glu_bwd!(P, da, db, bias, mask, t0::Int32, n::Int32, lda::Int32,
    mmod::Int32, ::Val{H}, ::Val{NTE}, ::Val{HB}, ::Val{HM}) where {H,NTE,HB,HM}
    i = (_i32(@index(Group, Linear)) - Int32(1)) * Int32(NTE) + _i32(@index(Local, Linear)) - Int32(1)
    if i < n * Int32(H)
        tt = _srem(i, n); h = _sdiv(i, n); t = t0 + tt
        keep = HM ? unsafe_load(mask, _srem(t, mmod) + 1) : true
        pn = Int64(n)
        p0 = Int64(tt) + 1
        av = Float32(unsafe_load(P, p0 + pn * h))
        bv = Float32(unsafe_load(P, p0 + pn * (Int32(H) + h)))
        ag = Float32(unsafe_load(P, p0 + pn * (Int32(2H) + h)))
        bg = Float32(unsafe_load(P, p0 + pn * (Int32(3H) + h)))
        if HB
            av += Float32(unsafe_load(bias, h + 1))
            bv += Float32(unsafe_load(bias, Int32(H) + h + 1))
            ag += Float32(unsafe_load(bias, Int32(2H) + h + 1))
            bg += Float32(unsafe_load(bias, Int32(3H) + h + 1))
        end
        o = Int64(t) + Int64(lda) * h + 1
        ga = keep ? Float32(unsafe_load(da, o)) : 0.0f0
        gb = keep ? Float32(unsafe_load(db, o)) : 0.0f0
        sa = _sigmoid(ag); sb = _sigmoid(bg)
        unsafe_store!(P, eltype(P)(ga * sa), p0 + pn * h)
        unsafe_store!(P, eltype(P)(gb * sb), p0 + pn * (Int32(H) + h))
        unsafe_store!(P, eltype(P)(ga * av * sa * (1.0f0 - sa)), p0 + pn * (Int32(2H) + h))
        unsafe_store!(P, eltype(P)(gb * bv * sb * (1.0f0 - sb)), p0 + pn * (Int32(3H) + h))
    end
end

"""
    trimul_ln_bwd!(dx, x, dy, dy2, γ, part, t0, n, ldx, dyoff, eps, Val(D), Val(TB), Val(NT), Val(TOKMAJ), Val(DY2))

The backward of [`trimul_ln!`](@ref): the cotangent `dx` of the LayerNorm's input for the `n` tokens `t0 + 1 … t0 + n`
of `x`, from the cotangent `dy` of its output. `x` and `dx` have the same layout, channel-first (`TOKMAJ = false`) or
token-major with leading dimension `ldx` (`true`); `dx` may be `x` itself or `dy`'s buffer (every tile is read before
anything is written). `dy` is channel-first `[D, ·]`, token `t` of the block at column `dyoff + t`; with `DY2`, the
block `dy2[:, 1:n]` (also channel-first) is added to it, so a cotangent that arrives in two parts needs no pass of its
own. With `g = γ·dy`
and `x̂` the normalised input, `dx = rstd·(g − mean(g) − x̂·mean(g·x̂))`. The statistics are recomputed from `x` exactly
as the forward does. Besides `dx`, a group writes its partial sums to `part[:, group]` (`2D` of them): `Σ_tokens dy·x̂`
(the scale's gradient) in rows `1:D` and `Σ_tokens dy` (the bias's) in rows `D+1:2D`; the caller sums the groups
(`mul!` with ones: a fixed order, no atomics).
"""
@kernel unsafe_indices = true function trimul_ln_bwd!(dx, x, dy, dy2, γ, part, t0::Int32, n::Int32, ldx::Int32,
    dyoff::Int32, eps::Float32, ::Val{D}, ::Val{TB}, ::Val{NT}, ::Val{TOKMAJ}, ::Val{DY2}) where {D,TB,NT,TOKMAJ,DY2}
    xt = @localmem Float32 (D + 1, TB)
    dt = @localmem Float32 (D + 1, TB)
    part1 = @localmem Float32 (NT ÷ TB, TB)
    part2 = @localmem Float32 (NT ÷ TB, TB)
    stat = @localmem Float32 (4, TB)

    tid = _i32(@index(Local, Linear)) - Int32(1)
    tb0 = (_i32(@index(Group, Linear)) - Int32(1)) * Int32(TB)
    e = tid
    while e < Int32(D * TB)
        if TOKMAJ
            tt = _srem(e, Int32(TB)); d = _sdiv(e, Int32(TB))
        else
            d = _srem(e, Int32(D)); tt = _sdiv(e, Int32(D))
        end
        t = tb0 + tt
        v = 0.0f0
        if t < n
            g = Int64(t0 + t)
            v = Float32(TOKMAJ ? unsafe_load(x, g + Int64(ldx) * d + 1) : unsafe_load(x, Int64(D) * g + d + 1))
        end
        @inbounds xt[d + 1, tt + 1] = v
        e += Int32(NT)
    end
    e = tid
    while e < Int32(D * TB)
        d = _srem(e, Int32(D)); tt = _sdiv(e, Int32(D)); t = tb0 + tt
        v = 0.0f0
        if t < n
            v = Float32(unsafe_load(dy, Int64(D) * Int64(dyoff + t) + d + 1))
            if DY2
                v += Float32(unsafe_load(dy2, Int64(D) * Int64(t) + d + 1))
            end
        end
        @inbounds dt[d + 1, tt + 1] = v
        e += Int32(NT)
    end
    @synchronize

    tid = _i32(@index(Local, Linear)) - Int32(1)
    tok = _srem(tid, Int32(TB)); p = _sdiv(tid, Int32(TB))
    s = 0.0f0
    d = p
    while d < Int32(D)
        s += @inbounds xt[d + 1, tok + 1]
        d += Int32(NT ÷ TB)
    end
    @inbounds part1[p + 1, tok + 1] = s
    @synchronize

    tid = _i32(@index(Local, Linear)) - Int32(1)
    if tid < Int32(TB)
        s = 0.0f0
        for q in 1:(NT ÷ TB)
            s += @inbounds part1[q, tid + 1]
        end
        @inbounds stat[1, tid + 1] = s / Float32(D)
    end
    @synchronize

    tid = _i32(@index(Local, Linear)) - Int32(1)
    tok = _srem(tid, Int32(TB)); p = _sdiv(tid, Int32(TB))
    μ = @inbounds stat[1, tok + 1]
    s = 0.0f0
    d = p
    while d < Int32(D)
        δ = @inbounds(xt[d + 1, tok + 1]) - μ
        s += δ * δ
        d += Int32(NT ÷ TB)
    end
    @inbounds part1[p + 1, tok + 1] = s
    @synchronize

    tid = _i32(@index(Local, Linear)) - Int32(1)
    if tid < Int32(TB)
        s = 0.0f0
        for q in 1:(NT ÷ TB)
            s += @inbounds part1[q, tid + 1]
        end
        @inbounds stat[2, tid + 1] = inv(sqrt(s / Float32(D) + eps))
    end
    @synchronize

    # Σ g and Σ g·x̂ per token, g = γ·dy.
    tid = _i32(@index(Local, Linear)) - Int32(1)
    tok = _srem(tid, Int32(TB)); p = _sdiv(tid, Int32(TB))
    μ = @inbounds stat[1, tok + 1]
    rs = @inbounds stat[2, tok + 1]
    s1 = 0.0f0
    s2 = 0.0f0
    d = p
    while d < Int32(D)
        gd = Float32(unsafe_load(γ, d + 1)) * @inbounds(dt[d + 1, tok + 1])
        s1 += gd
        s2 += gd * (@inbounds(xt[d + 1, tok + 1]) - μ) * rs
        d += Int32(NT ÷ TB)
    end
    @inbounds part1[p + 1, tok + 1] = s1
    @inbounds part2[p + 1, tok + 1] = s2
    @synchronize

    tid = _i32(@index(Local, Linear)) - Int32(1)
    if tid < Int32(TB)
        s1 = 0.0f0
        s2 = 0.0f0
        for q in 1:(NT ÷ TB)
            s1 += @inbounds part1[q, tid + 1]
            s2 += @inbounds part2[q, tid + 1]
        end
        @inbounds stat[3, tid + 1] = s1 / Float32(D)
        @inbounds stat[4, tid + 1] = s2 / Float32(D)
    end
    @synchronize

    # dx, written in the input's layout; each element's tile cell then holds dy·x̂ for the scale's gradient.
    tid = _i32(@index(Local, Linear)) - Int32(1)
    tb0 = (_i32(@index(Group, Linear)) - Int32(1)) * Int32(TB)
    e = tid
    while e < Int32(D * TB)
        if TOKMAJ
            tt = _srem(e, Int32(TB)); d = _sdiv(e, Int32(TB))
        else
            d = _srem(e, Int32(D)); tt = _sdiv(e, Int32(D))
        end
        t = tb0 + tt
        μ = @inbounds stat[1, tt + 1]
        rs = @inbounds stat[2, tt + 1]
        xh = (@inbounds(xt[d + 1, tt + 1]) - μ) * rs
        dyv = @inbounds dt[d + 1, tt + 1]
        if t < n
            v = rs * (Float32(unsafe_load(γ, d + 1)) * dyv - @inbounds(stat[3, tt + 1]) - xh * @inbounds(stat[4, tt + 1]))
            g = Int64(t0 + t)
            if TOKMAJ
                unsafe_store!(dx, eltype(dx)(v), g + Int64(ldx) * d + 1)
            else
                unsafe_store!(dx, eltype(dx)(v), Int64(D) * g + d + 1)
            end
        end
        @inbounds xt[d + 1, tt + 1] = dyv * xh
        e += Int32(NT)
    end
    @synchronize

    tid = _i32(@index(Local, Linear)) - Int32(1)
    grp = Int64(_i32(@index(Group, Linear)) - Int32(1))
    d = tid
    while d < Int32(D)
        s = 0.0f0
        sb = 0.0f0
        for tt in 1:TB
            s += @inbounds xt[d + 1, tt]
            sb += @inbounds dt[d + 1, tt]
        end
        unsafe_store!(part, s, Int64(2D) * grp + d + 1)
        unsafe_store!(part, sb, Int64(2D) * grp + D + d + 1)
        d += Int32(NT)
    end
end

"""
    trimul_argtypes(stage, C, H, p, hb, hm, T=Float32, Ta=T) -> Tuple

The argument types of one stage's binary, in slot order (the buffers as `AbstractArray{T}`, then the scalars,
then the compile-time `Val`s), for `ka_compile`. `p` holds the tunables (`trimul_tunables`). `T` is the storage type
of the data and parameters (`Float32`, `Float16`, `BFloat16`), `Ta` that of `a`, `b` and the contraction result `X`:
`T`, or `Float32` for 16-bit data on the [`HalfSafe`](@ref) path, whose contraction runs in Float32. All arithmetic in
the kernels is `Float32`. The binding order is [`trimul_args`](@ref)'s, pinned by `test/kernels/trimul_op.test.jl`.
"""
function trimul_argtypes(stage::Symbol, C::Int, H::Int, p, hb::Bool, hm::Bool, ::Type{T}=Float32, ::Type{Ta}=T) where {T,Ta}
    F, A, AA = Float32, AbstractArray{T}, AbstractArray{Ta}
    return stage === :ln_in ? (A, A, A, A, Int32, Int32, Int32, F, Val{C}, Val{p.TB}, Val{p.NT}, Val{false}) :
           stage === :ln_out ? (A, AA, A, A, Int32, Int32, Int32, F, Val{H}, Val{p.TB}, Val{p.NT}, Val{true}) :
           stage === :glu ? (AA, AA, A, A, AbstractArray{Bool}, Int32, Int32, Int32, Int32,
        Val{H}, Val{p.NTE}, Val{hb}, Val{hm}) :
           stage === :gate ? (A, A, A, A, A, Int32, Int32, Val{C}, Val{p.NTE}, Val{hb}) :
           stage === :gate_bwd ? (A, A, A, A, A, Int32, Int32, Val{C}, Val{p.NTE}, Val{hb}) :
           stage === :glu_bwd ? (A, AA, AA, A, AbstractArray{Bool}, Int32, Int32, Int32, Int32,
        Val{H}, Val{p.NTE}, Val{hb}, Val{hm}) :
           stage === :ln_in_bwd ? (A, A, A, A, A, A, Int32, Int32, Int32, Int32, F,
        Val{C}, Val{p.TB}, Val{p.NT}, Val{false}, Val{true}) :
           stage === :ln_out_bwd ? (AA, AA, A, A, A, A, Int32, Int32, Int32, Int32, F,
        Val{H}, Val{p.TB}, Val{p.NT}, Val{true}, Val{false}) :
           throw(ArgumentError("no TriMul stage `:$stage`; stages are $(TRIMUL_STAGES) and $(TRIMUL_BWD_STAGES)"))
end

const TRIMUL_STAGES = (:ln_in, :glu, :ln_out, :gate)
const TRIMUL_BWD_STAGES = (:gate_bwd, :glu_bwd, :ln_out_bwd, :ln_in_bwd)
const _LN_STAGES = (:ln_in, :ln_out, :ln_in_bwd, :ln_out_bwd)
trimul_stage_kernel(stage::Symbol) =
    stage in (:ln_in, :ln_out) ? trimul_ln! : stage in (:ln_in_bwd, :ln_out_bwd) ? trimul_ln_bwd! :
    stage === :glu ? trimul_glu! : stage === :gate ? trimul_gate! :
    stage === :glu_bwd ? trimul_glu_bwd! : stage === :gate_bwd ? trimul_gate_bwd! :
    throw(ArgumentError("no TriMul stage `:$stage`"))
# Threads per group, and what one group covers (the binary's `tile`): a LayerNorm group covers TB tokens, an
# elementwise group NTE elements.
trimul_stage_threads(stage::Symbol, p) = stage in _LN_STAGES ? p.NT : p.NTE
trimul_stage_tile(stage::Symbol, p) = stage in _LN_STAGES ? p.TB : p.NTE
