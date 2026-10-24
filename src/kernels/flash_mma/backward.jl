# ---------------------------------------------------------------------------------------------
# Backward, dK / dV / dbias: rows are KEYS
# ---------------------------------------------------------------------------------------------

# One query tile for the R key fragments of this simdgroup. The scores are formed as `Sᵀ = K·Qᵀ` in blocks of `FN`
# queries; `FK÷FN` blocks make one chunk, the contraction axis of `dV += Pᵀ·dO` and `dK += dSᵀ·Q`.
@generated function _mma_kv_tile(be, dK, dV, Kf, Vf, qT, qN, dT, dN, ls, dl, bias, dbias, live,
    keys, bh, n, t, col, ::Val{R}, ::Val{DIM}, ::Val{BQ}, ::Val{HB}) where {R,DIM,BQ,HB}
    FM, FN, FK = _mma_dims(be)
    NR, NKD, NQK, NDN, NB = FM ÷ 8, DIM ÷ FK, BQ ÷ FK, DIM ÷ FN, FK ÷ FN
    body = Expr[]
    dk = [Symbol(:dk_, r, :_, d) for r in 1:R, d in 1:NDN]
    dv = [Symbol(:dv_, r, :_, d) for r in 1:R, d in 1:NDN]
    for r in 1:R, d in 1:NDN
        push!(body, :($(dk[r, d]) = dK[$r][$d]), :($(dv[r, d]) = dV[$r][$d]))
    end
    # per slot `u` and block `j` of a chunk: the probabilities and score gradients of this lane's two queries
    P1(u, j) = Symbol(:p1_, u, :_, j)
    P2(u, j) = Symbol(:p2_, u, :_, j)
    D1(u, j) = Symbol(:ds1_, u, :_, j)
    D2(u, j) = Symbol(:ds2_, u, :_, j)
    for qc in 1:NQK
        for j in 0:NB-1
            qb = NB * (qc - 1) + j
            qt = [Symbol(:qt_, k) for k in 1:NKD]
            dt = [Symbol(:dt_, k) for k in 1:NKD]
            for k in 1:NKD
                push!(body, :($(qt[k]) = _mma_load_b(be, qT, $(FK * (k - 1) + 1), $(FN * qb + 1))),
                    :($(dt[k]) = _mma_load_b(be, dT, $(FK * (k - 1) + 1), $(FN * qb + 1))))
            end
            for r in 1:R
                sx = :(_mma_zero(be, Float32))
                px = :(_mma_zero(be, Float32))
                for k in 1:NKD
                    sx = :(_mma(be, Kf[$r][$k], $(qt[k]), $sx))
                    px = :(_mma(be, Vf[$r][$k], $(dt[k]), $px))
                end
                push!(body, :(sacc = $sx), :(pacc = $px))
                for s in 1:NR
                    u = NR * (r - 1) + s
                    push!(body, quote
                        s1, s2 = _mma_pair(be, sacc, $s)
                        dp1, dp2 = _mma_pair(be, pacc, $s)
                        c = Int32($(FN * qb)) + col        # query (within the tile) of s1
                        key = keys + Int32($(8 * (u - 1)))
                        kc = min(key, n - Int32(1))
                        if HB
                            q1 = min(t + c, n - Int32(1))
                            q2 = min(t + c + Int32(1), n - Int32(1))
                            s1 += Float32(unsafe_load(bias, (bh * n + q1) * n + kc + Int32(1))) * _LOG2E
                            s2 += Float32(unsafe_load(bias, (bh * n + q2) * n + kc + Int32(1))) * _LOG2E
                        end
                        lv = live[$u]
                        # Dead keys contribute exactly zero: SELECTED, not multiplied, so a NaN `delta` (a
                        # fully masked query row) cannot leak. Queries past `n` have lse = +Inf: p = 0.
                        p1 = lv ? exp2(s1 - @inbounds(ls[c+Int32(1)])) : 0f0
                        p2 = lv ? exp2(s2 - @inbounds(ls[c+Int32(2)])) : 0f0
                        ds1 = lv ? p1 * (dp1 - @inbounds(dl[c+Int32(1)])) : 0f0
                        ds2 = lv ? p2 * (dp2 - @inbounds(dl[c+Int32(2)])) : 0f0
                        if HB && lv
                            t + c < n &&
                                _atomic_fadd!(dbias, (bh * n + t + c) * n + key + Int32(1), ds1)
                            t + c + Int32(1) < n &&
                                _atomic_fadd!(dbias, (bh * n + t + c + Int32(1)) * n + key + Int32(1), ds2)
                        end
                        $(P1(u, j)), $(P2(u, j)) = p1, p2
                        $(D1(u, j)), $(D2(u, j)) = ds1, ds2
                    end)
                end
            end
        end
        # P and dS as the operands of the next products, one per chunk of `FK` queries.
        for r in 1:R
            ps = [:(($(P1(NR * (r - 1) + s, j)), $(P2(NR * (r - 1) + s, j)))) for j in 0:NB-1 for s in 1:NR]
            ds = [:(($(D1(NR * (r - 1) + s, j)), $(D2(NR * (r - 1) + s, j)))) for j in 0:NB-1 for s in 1:NR]
            push!(body, :($(Symbol(:pf_, r)) = _mma_pfrag(be, eltype(qN), $(ps...))),
                :($(Symbol(:sf_, r)) = _mma_pfrag(be, eltype(qN), $(ds...))))
        end
        for d in 1:NDN
            push!(body, :(dnf = _mma_load_b(be, dN, $(FK * (qc - 1) + 1), $(FN * (d - 1) + 1))),
                :(qnf = _mma_load_b(be, qN, $(FK * (qc - 1) + 1), $(FN * (d - 1) + 1))))
            for r in 1:R
                push!(body, :($(dv[r, d]) = _mma(be, $(Symbol(:pf_, r)), dnf, $(dv[r, d]))),
                    :($(dk[r, d]) = _mma(be, $(Symbol(:sf_, r)), qnf, $(dk[r, d]))))
            end
        end
    end
    tup(m) = Expr(:tuple, (Expr(:tuple, (m[r, d] for d in 1:NDN)...) for r in 1:R)...)
    return Expr(:block, Expr(:meta, :inline), body..., :(($(tup(dk)), $(tup(dv)))))
end

@inline _mma_live(mask, mb, keys, n, ::Val{U}, ::Val{HM}) where {U,HM} = ntuple(Val(U)) do u
    key = keys + Int32(8 * (u - 1))
    key < n && !(HM && unsafe_load(mask, mb + key + Int32(1)))
end

"""
    flash_mma_bwd_kv!(dk, dv, dbias, q, k, v, bias, mask, dO, lse, delta, scale, n, nouter, nh,
        mheads, Val(BE), Val(T), Val(DIM), Val(NSG), Val(R), Val(BQ), Val(HB), Val(HM), Val(PE))

The `dK`/`dV`/`dbias` stage of `:flash_mma`. Rows are KEYS: each simdgroup holds `R` fragments of `FM`
keys' `K` and `V` operands and walks the queries in tiles of `BQ`, forming `Sᵀ = K·Qᵀ` and
`dPᵀ = V·dOᵀ`, then `dV += Pᵀ·dO` and `dK += dSᵀ·Q`. `dbias` (`Float32`, keys-fastest, ZEROED by the
caller) is accumulated with atomics, because the bias has no outer axis. `lse` and `delta` are the
forward's residual and `Σ o·dO`, both `Float32`, laid out as `flash_fwd!`'s `lse`.
"""
@kernel unsafe_indices = true function flash_mma_bwd_kv!(dk, dv, dbias, q, k, v, bias, mask, dO,
    lse, delta, scale, n, nouter, nh, mheads, ::Val{BE}, ::Val{T}, ::Val{DIM}, ::Val{NSG},
    ::Val{R}, ::Val{BQ}, ::Val{HB}, ::Val{HM}, ::Val{PE}) where {BE,T,DIM,NSG,R,BQ,HB,HM,PE}
    be = BE()
    FM, FN, FK = _mma_shape(be)
    U = R * (FM ÷ 8)                       # row slots (keys) this lane holds
    BKV = FM * R * NSG
    qT = @localmem T (DIM, BQ)
    dT = @localmem T (DIM, BQ)
    qN = @localmem PE (BQ, DIM)         # B operands of dS·Q and P·dO, against P/dS in `PE`
    dN = @localmem PE (BQ, DIM)
    sd = @localmem Float32 (DIM,)       # a dead slice's running Σⱼ dO[j], always Float32
    ls = @localmem Float32 (BQ,)
    dl = @localmem Float32 (BQ,)
    kst = @localmem T (BKV, DIM)

    NT = Int32(32 * NSG)
    g = _i32(@index(Group, Linear)) - Int32(1)
    tid = _i32(@index(Local, Linear)) - Int32(1)
    sg = tid ÷ Int32(32)
    lane = tid % Int32(32)
    nkt = (n + Int32(BKV) - Int32(1)) ÷ Int32(BKV)
    kt = _srem(g, nkt)
    rest = _sdiv(g, nkt)
    i = _srem(rest, nouter)
    bh = _sdiv(rest, nouter)
    hi = _srem(bh, nh)
    bi = _sdiv(bh, nh)
    row = (bi * nouter + i) * n
    step = nh * Int32(DIM)
    base = row * step + hi * Int32(DIM)
    k0 = kt * Int32(BKV)
    r0 = sg * Int32(FM * R)
    fr, col = _mma_lane(be, lane)
    keys = k0 + r0 + fr
    lrow = ((bi * nouter + i) * nh + hi) * n

    # A FULLY MASKED slice (`flash_mma_fwd!` wrote `lse = -Inf` for every query of it) attended uniformly, so
    # `dV[key] = Σⱼ dO[j] / n` for EVERY key < n, and `dK` is zero (the logits were replaced, `dbias` untouched).
    # Whether the slice is dead is one `lse` load, the same for every thread of the group, and the branch is
    # taken once, outside both tile loops: a live slice runs exactly the code below it. The running Σⱼ dO[j] is kept in `sd` (Float32: `qN` may be 16-bit). Staging `dO` into `dN` is
    # exact, since `PE` is the storage type or Float32.
    if unsafe_load(lse, lrow + Int32(1)) == -Inf32
        e = tid
        while e < Int32(DIM)
            @inbounds sd[e+Int32(1)] = 0f0
            e += NT
        end
        t = Int32(0)
        while t < n
            @synchronize
            e = tid
            while e < Int32(BQ * DIM)
                d = e % Int32(DIM); c = e ÷ Int32(DIM); j = t + c
                @inbounds dN[c+Int32(1), d+Int32(1)] =
                    j < n ? PE(Float32(unsafe_load(dO, base + j * step + d + Int32(1)))) : zero(PE)
                e += NT
            end
            @synchronize
            e = tid
            while e < Int32(DIM)
                acc = @inbounds sd[e+Int32(1)]
                for c in Int32(1):Int32(BQ)
                    acc += Float32(@inbounds dN[c, e+Int32(1)])
                end
                @inbounds sd[e+Int32(1)] = acc
                e += NT
            end
            t += Int32(BQ)
        end
        @synchronize
        invn = inv(Float32(n))
        e = tid
        while e < Int32(BKV * DIM)
            d = e % Int32(DIM); r = e ÷ Int32(DIM); kk = k0 + r
            if kk < n
                off = base + kk * step + d + Int32(1)
                unsafe_store!(dv, T(@inbounds(sd[d+Int32(1)]) * invn), off)
                unsafe_store!(dk, zero(T), off)
            end
            e += NT
        end
    else
        # K, then V, of this group's keys into registers, through one staging buffer.
        e = tid
        while e < Int32(BKV * DIM)
            d = e % Int32(DIM); r = e ÷ Int32(DIM); kk = k0 + r
            @inbounds kst[r+Int32(1), d+Int32(1)] = kk < n ? unsafe_load(k, base + kk * step + d + Int32(1)) : zero(T)
            e += NT
        end
        @synchronize
        Kf = _mma_rows(be, kst, r0, Val(R), Val(DIM))
        @synchronize
        e = tid
        while e < Int32(BKV * DIM)
            d = e % Int32(DIM); r = e ÷ Int32(DIM); kk = k0 + r
            @inbounds kst[r+Int32(1), d+Int32(1)] = kk < n ? unsafe_load(v, base + kk * step + d + Int32(1)) : zero(T)
            e += NT
        end
        @synchronize
        Vf = _mma_rows(be, kst, r0, Val(R), Val(DIM))
        live = _mma_live(mask, (_sdiv(bh, mheads) * nouter + i) * n, keys, n, Val(U), Val(HM))
        dK = _mma_zeros(be, Val(R), Val(DIM))
        dV = _mma_zeros(be, Val(R), Val(DIM))

        t = Int32(0)
        while t < n
            @synchronize
            e = tid
            while e < Int32(BQ * DIM)
                d = e % Int32(DIM); c = e ÷ Int32(DIM); j = t + c
                off = base + j * step + d + Int32(1)
                xq = j < n ? Float32(unsafe_load(q, off)) * scale * _LOG2E : 0f0
                xd = j < n ? Float32(unsafe_load(dO, off)) : 0f0
                @inbounds qT[d+Int32(1), c+Int32(1)] = T(xq)
                @inbounds qN[c+Int32(1), d+Int32(1)] = PE(xq)
                @inbounds dT[d+Int32(1), c+Int32(1)] = T(xd)
                @inbounds dN[c+Int32(1), d+Int32(1)] = PE(xd)
                e += NT
            end
            e = tid
            while e < Int32(BQ)
                j = t + e
                @inbounds ls[e+Int32(1)] = j < n ? unsafe_load(lse, lrow + j + Int32(1)) * _LOG2E : Inf32
                @inbounds dl[e+Int32(1)] = j < n ? unsafe_load(delta, lrow + j + Int32(1)) : 0f0
                e += NT
            end
            @synchronize
            dK, dV = _mma_kv_tile(be, dK, dV, Kf, Vf, qT, qN, dT, dN, ls, dl, bias, dbias, live, keys,
                bh, n, t, col, Val(R), Val(DIM), Val(BQ), Val(HB))
            t += Int32(BQ)
        end
        # qN held q·scale·log2e, so dK = scale·Σ dS·q is the accumulator over log2e.
        _mma_store_rows!(be, dk, dK, inv(_LOG2E), base, step, keys, n, col, Val(R), Val(DIM), T)
        _mma_store_rows!(be, dv, dV, 1f0, base, step, keys, n, col, Val(R), Val(DIM), T)
    end
end

# ---------------------------------------------------------------------------------------------
# Backward, dQ: rows are QUERIES
# ---------------------------------------------------------------------------------------------

# One key tile for the R query fragments of this simdgroup: `S = Q·Kᵀ` and `dP = dO·Vᵀ` in blocks of `FN` keys,
# `dQ += dS·K` over chunks of `FK` keys.
@generated function _mma_q_tile(be, dQ, Qf, Df, ks, kn, vT, ms, bias, rs, bbs, t, n, col, ::Val{R},
    ::Val{DIM}, ::Val{BK}, ::Val{HB}) where {R,DIM,BK,HB}
    FM, FN, FK = _mma_dims(be)
    NR, NKD, NKK, NDN, NB = FM ÷ 8, DIM ÷ FK, BK ÷ FK, DIM ÷ FN, FK ÷ FN
    body = Expr[]
    dq = [Symbol(:dq_, r, :_, d) for r in 1:R, d in 1:NDN]
    for r in 1:R, d in 1:NDN
        push!(body, :($(dq[r, d]) = dQ[$r][$d]))
    end
    D1(u, j) = Symbol(:ds1_, u, :_, j)
    D2(u, j) = Symbol(:ds2_, u, :_, j)
    for kc in 1:NKK
        for j in 0:NB-1
            kb = NB * (kc - 1) + j
            kf = [Symbol(:kf_, k) for k in 1:NKD]
            vf = [Symbol(:vf_, k) for k in 1:NKD]
            for k in 1:NKD
                push!(body, :($(kf[k]) = _mma_load_b(be, ks, $(FK * (k - 1) + 1), $(FN * kb + 1))),
                    :($(vf[k]) = _mma_load_b(be, vT, $(FK * (k - 1) + 1), $(FN * kb + 1))))
            end
            for r in 1:R
                sx = :(_mma_zero(be, Float32))
                px = :(_mma_zero(be, Float32))
                for k in 1:NKD
                    sx = :(_mma(be, Qf[$r][$k], $(kf[k]), $sx))
                    px = :(_mma(be, Df[$r][$k], $(vf[k]), $px))
                end
                push!(body, :(sacc = $sx), :(pacc = $px))
                for s in 1:NR
                    u = NR * (r - 1) + s
                    push!(body, quote
                        s1, s2 = _mma_pair(be, sacc, $s)
                        dp1, dp2 = _mma_pair(be, pacc, $s)
                        c = Int32($(FN * kb)) + col
                        live1 = @inbounds(ms[c+Int32(1)]) == 0f0
                        live2 = @inbounds(ms[c+Int32(2)]) == 0f0
                        b1, b2 = _mma_bias2(bias, bbs[$u], t + c, n, Val(HB))
                        l2, dlt = rs[$u]
                        p1 = live1 ? exp2(s1 + b1 - l2) : 0f0
                        p2 = live2 ? exp2(s2 + b2 - l2) : 0f0
                        $(D1(u, j)) = live1 ? p1 * (dp1 - dlt) : 0f0
                        $(D2(u, j)) = live2 ? p2 * (dp2 - dlt) : 0f0
                    end)
                end
            end
        end
        for r in 1:R
            ds = [:(($(D1(NR * (r - 1) + s, j)), $(D2(NR * (r - 1) + s, j)))) for j in 0:NB-1 for s in 1:NR]
            push!(body, :($(Symbol(:sf_, r)) = _mma_pfrag(be, eltype(kn), $(ds...))))
        end
        for d in 1:NDN
            push!(body, :(knf = _mma_load_b(be, kn, $(FK * (kc - 1) + 1), $(FN * (d - 1) + 1))))
            for r in 1:R
                push!(body, :($(dq[r, d]) = _mma(be, $(Symbol(:sf_, r)), knf, $(dq[r, d]))))
            end
        end
    end
    return Expr(:block, Expr(:meta, :inline), body...,
        Expr(:tuple, (Expr(:tuple, (dq[r, d] for d in 1:NDN)...) for r in 1:R)...))
end

@inline _mma_row_stats(lse, delta, lrow, j0, n, ::Val{U}) where {U} = ntuple(Val(U)) do u
    j = min(j0 + Int32(8 * (u - 1)), n - Int32(1))
    (unsafe_load(lse, lrow + j + Int32(1)) * _LOG2E, unsafe_load(delta, lrow + j + Int32(1)))
end

"""
    flash_mma_bwd_q!(dq, q, k, v, bias, mask, dO, lse, delta, scale, n, nouter, nh, mheads,
        Val(BE), Val(T), Val(DIM), Val(NSG), Val(R), Val(BK), Val(HB), Val(HM), Val(PE))

The `dQ` stage of `:flash_mma`: rows are queries, holding `Q` and `dO` fragments; per key tile,
`S = Q·Kᵀ`, `dP = dO·Vᵀ`, `dS = P∘(dP − delta)`, `dQ += dS·K`.
"""
@kernel unsafe_indices = true function flash_mma_bwd_q!(dq, q, k, v, bias, mask, dO, lse, delta,
    scale, n, nouter, nh, mheads, ::Val{BE}, ::Val{T}, ::Val{DIM}, ::Val{NSG},
    ::Val{R}, ::Val{BK}, ::Val{HB}, ::Val{HM}, ::Val{PE}) where {BE,T,DIM,NSG,R,BK,HB,HM,PE}
    be = BE()
    FM, FN, FK = _mma_shape(be)
    U = R * (FM ÷ 8)                       # row slots (queries) this lane holds
    BQ = FM * R * NSG
    qs = @localmem T (BQ, DIM)
    ks = @localmem T (DIM, BK)     # Kᵀ for S = Q·Kᵀ
    vT = @localmem T (DIM, BK)     # Vᵀ for dP = dO·Vᵀ
    kn = @localmem PE (BK, DIM)    # K for dQ = dS·K, against dS in `PE`
    ms = @localmem Float32 (BK,)

    NT = Int32(32 * NSG)
    g = _i32(@index(Group, Linear)) - Int32(1)
    tid = _i32(@index(Local, Linear)) - Int32(1)
    sg = tid ÷ Int32(32)
    lane = tid % Int32(32)
    nqt = (n + Int32(BQ) - Int32(1)) ÷ Int32(BQ)
    qt = _srem(g, nqt)
    rest = _sdiv(g, nqt)
    i = _srem(rest, nouter)
    bh = _sdiv(rest, nouter)
    hi = _srem(bh, nh)
    bi = _sdiv(bh, nh)
    row = (bi * nouter + i) * n
    step = nh * Int32(DIM)
    base = row * step + hi * Int32(DIM)
    q0 = qt * Int32(BQ)
    r0 = sg * Int32(FM * R)
    fr, col = _mma_lane(be, lane)
    j0 = q0 + r0 + fr
    lrow = ((bi * nouter + i) * nh + hi) * n
    mb = (_sdiv(bh, mheads) * nouter + i) * n

    e = tid
    while e < Int32(BQ * DIM)
        d = e % Int32(DIM); r = e ÷ Int32(DIM); j = q0 + r
        x = j < n ? Float32(unsafe_load(q, base + j * step + d + Int32(1))) * scale * _LOG2E : 0f0
        @inbounds qs[r+Int32(1), d+Int32(1)] = T(x)
        e += NT
    end
    @synchronize
    Qf = _mma_rows(be, qs, r0, Val(R), Val(DIM))
    @synchronize
    e = tid
    while e < Int32(BQ * DIM)
        d = e % Int32(DIM); r = e ÷ Int32(DIM); j = q0 + r
        @inbounds qs[r+Int32(1), d+Int32(1)] = j < n ? unsafe_load(dO, base + j * step + d + Int32(1)) : zero(T)
        e += NT
    end
    @synchronize
    Df = _mma_rows(be, qs, r0, Val(R), Val(DIM))
    dQ = _mma_zeros(be, Val(R), Val(DIM))
    rs = _mma_row_stats(lse, delta, lrow, j0, n, Val(U))
    bbs = _mma_bias_rows(bh, n, j0, Val(U))

    t = Int32(0)
    while t < n
        @synchronize
        e = tid
        while e < Int32(DIM * BK)
            d = e % Int32(DIM); c = e ÷ Int32(DIM); kk = t + c
            off = base + kk * step + d + Int32(1)
            xk = kk < n ? unsafe_load(k, off) : zero(T)
            @inbounds ks[d+Int32(1), c+Int32(1)] = xk
            @inbounds kn[c+Int32(1), d+Int32(1)] = PE(Float32(xk))
            @inbounds vT[d+Int32(1), c+Int32(1)] = kk < n ? unsafe_load(v, off) : zero(T)
            e += NT
        end
        e = tid
        while e < Int32(BK)
            kk = t + e
            dead = kk < n ? (HM ? unsafe_load(mask, mb + kk + Int32(1)) : false) : true
            @inbounds ms[e+Int32(1)] = dead ? 1f0 : 0f0
            e += NT
        end
        @synchronize
        dQ = _mma_q_tile(be, dQ, Qf, Df, ks, kn, vT, ms, bias, rs, bbs, t, n, col, Val(R),
            Val(DIM), Val(BK), Val(HB))
        t += Int32(BK)
    end
    _mma_store_rows!(be, dq, dQ, scale, base, step, j0, n, col, Val(R), Val(DIM), T)
end
