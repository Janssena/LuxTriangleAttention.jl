# ---------------------------------------------------------------------------------------------
# Backward, dK / dV / dbias: rows are KEYS
# ---------------------------------------------------------------------------------------------

@generated function _mma_kv_tile(be, dK, dV, Kf, Vf, qT, qN, dT, dN, ls, dl, bias, dbias, live,
    keys, bh, n, t, col, ::Val{R}, ::Val{ND}, ::Val{NQB}, ::Val{HB}) where {R,ND,NQB,HB}
    body = Expr[]
    dk = [Symbol(:dk_, r, :_, d) for r in 1:R, d in 1:ND]
    dv = [Symbol(:dv_, r, :_, d) for r in 1:R, d in 1:ND]
    for r in 1:R, d in 1:ND
        push!(body, :($(dk[r, d]) = dK[$r][$d]), :($(dv[r, d]) = dV[$r][$d]))
    end
    for qb in 0:NQB-1
        qt = [Symbol(:qt_, d) for d in 1:ND]
        dt = [Symbol(:dt_, d) for d in 1:ND]
        for d in 1:ND
            push!(body, :($(qt[d]) = _mma_load(be, qT, $(8(d - 1) + 1), $(8qb + 1))),
                :($(dt[d]) = _mma_load(be, dT, $(8(d - 1) + 1), $(8qb + 1))))
        end
        for r in 1:R
            sx = :(_mma_zero(be, Float32))
            px = :(_mma_zero(be, Float32))
            for d in 1:ND
                sx = :(_mma(be, Kf[$r][$d], $(qt[d]), $sx))
                px = :(_mma(be, Vf[$r][$d], $(dt[d]), $px))
            end
            push!(body, quote
                s1, s2 = _mma_pair(be, $sx)
                dp1, dp2 = _mma_pair(be, $px)
                c = Int32($(8qb)) + col            # query (within the tile) of s1
                key = keys + Int32($(8(r - 1)))
                kc = min(key, n - Int32(1))
                if HB
                    q1 = min(t + c, n - Int32(1))
                    q2 = min(t + c + Int32(1), n - Int32(1))
                    s1 += Float32(unsafe_load(bias, (bh * n + q1) * n + kc + Int32(1))) * _LOG2E
                    s2 += Float32(unsafe_load(bias, (bh * n + q2) * n + kc + Int32(1))) * _LOG2E
                end
                lv = live[$r]
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
                $(Symbol(:pf_, r)) = _mma_frag(be, eltype(qN), p1, p2)
                $(Symbol(:sf_, r)) = _mma_frag(be, eltype(qN), ds1, ds2)
            end)
        end
        for d in 1:ND
            push!(body, :(dn = _mma_load(be, dN, $(8qb + 1), $(8(d - 1) + 1))),
                :(qn = _mma_load(be, qN, $(8qb + 1), $(8(d - 1) + 1))))
            for r in 1:R
                push!(body, :($(dv[r, d]) = _mma(be, $(Symbol(:pf_, r)), dn, $(dv[r, d]))),
                    :($(dk[r, d]) = _mma(be, $(Symbol(:sf_, r)), qn, $(dk[r, d]))))
            end
        end
    end
    tup(m) = Expr(:tuple, (Expr(:tuple, (m[r, d] for d in 1:ND)...) for r in 1:R)...)
    return Expr(:block, Expr(:meta, :inline), body..., :(($(tup(dk)), $(tup(dv)))))
end

@inline _mma_live(mask, mb, keys, n, ::Val{R}, ::Val{HM}) where {R,HM} = ntuple(Val(R)) do r
    key = keys + Int32(8 * (r - 1))
    key < n && !(HM && unsafe_load(mask, mb + key + Int32(1)))
end

"""
    flash_mma_bwd_kv!(dk, dv, dbias, q, k, v, bias, mask, dO, lse, delta, scale, n, nouter, nh,
        mheads, Val(BE), Val(T), Val(DIM), Val(NSG), Val(R), Val(BQ), Val(HB), Val(HM), Val(PE))

The `dK`/`dV`/`dbias` stage of `:flash_mma`. Rows are KEYS: each simdgroup holds `R` blocks of 8
keys' `K` and `V` fragments and walks the queries in tiles of `BQ`, forming `Sᵀ = K·Qᵀ` and
`dPᵀ = V·dOᵀ`, then `dV += Pᵀ·dO` and `dK += dSᵀ·Q`. `dbias` (`Float32`, keys-fastest, ZEROED by the
caller) is accumulated with atomics, because the bias has no outer axis. `lse` and `delta` are the
forward's residual and `Σ o·dO`, both `Float32`, laid out as `flash_fwd!`'s `lse`.
"""
@kernel unsafe_indices = true function flash_mma_bwd_kv!(dk, dv, dbias, q, k, v, bias, mask, dO,
    lse, delta, scale, n, nouter, nh, mheads, ::Val{BE}, ::Val{T}, ::Val{DIM}, ::Val{NSG},
    ::Val{R}, ::Val{BQ}, ::Val{HB}, ::Val{HM}, ::Val{PE}) where {BE,T,DIM,NSG,R,BQ,HB,HM,PE}
    be = BE()
    BKV = 8 * R * NSG
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
    kt = g % nkt
    rest = g ÷ nkt
    i = rest % nouter
    bh = rest ÷ nouter
    hi = bh % nh
    bi = bh ÷ nh
    row = (bi * nouter + i) * n
    step = nh * Int32(DIM)
    base = row * step + hi * Int32(DIM)
    k0 = kt * Int32(BKV)
    r0 = sg * Int32(8 * R)
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
        Kf = _mma_rows(be, kst, r0, Val(R), Val(DIM ÷ 8))
        @synchronize
        e = tid
        while e < Int32(BKV * DIM)
            d = e % Int32(DIM); r = e ÷ Int32(DIM); kk = k0 + r
            @inbounds kst[r+Int32(1), d+Int32(1)] = kk < n ? unsafe_load(v, base + kk * step + d + Int32(1)) : zero(T)
            e += NT
        end
        @synchronize
        Vf = _mma_rows(be, kst, r0, Val(R), Val(DIM ÷ 8))
        live = _mma_live(mask, ((bh ÷ mheads) * nouter + i) * n, keys, n, Val(R), Val(HM))
        dK = _mma_zeros(be, Val(R), Val(DIM ÷ 8))
        dV = _mma_zeros(be, Val(R), Val(DIM ÷ 8))

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
                bh, n, t, col, Val(R), Val(DIM ÷ 8), Val(BQ ÷ 8), Val(HB))
            t += Int32(BQ)
        end
        # qN held q·scale·log2e, so dK = scale·Σ dS·q is the accumulator over log2e.
        _mma_store_rows!(be, dk, dK, inv(_LOG2E), base, step, keys, n, col, Val(R), Val(DIM ÷ 8), T)
        _mma_store_rows!(be, dv, dV, 1f0, base, step, keys, n, col, Val(R), Val(DIM ÷ 8), T)
    end
end

# ---------------------------------------------------------------------------------------------
# Backward, dQ: rows are QUERIES
# ---------------------------------------------------------------------------------------------

@generated function _mma_q_tile(be, dQ, Qf, Df, ks, kn, vT, ms, bias, rs, bbs, t, n, col, ::Val{R},
    ::Val{ND}, ::Val{NKB}, ::Val{HB}) where {R,ND,NKB,HB}
    body = Expr[]
    dq = [Symbol(:dq_, r, :_, d) for r in 1:R, d in 1:ND]
    for r in 1:R, d in 1:ND
        push!(body, :($(dq[r, d]) = dQ[$r][$d]))
    end
    for kb in 0:NKB-1
        kf = [Symbol(:kf_, d) for d in 1:ND]
        vf = [Symbol(:vf_, d) for d in 1:ND]
        for d in 1:ND
            push!(body, :($(kf[d]) = _mma_load(be, ks, $(8(d - 1) + 1), $(8kb + 1))),
                :($(vf[d]) = _mma_load(be, vT, $(8(d - 1) + 1), $(8kb + 1))))
        end
        for r in 1:R
            sx = :(_mma_zero(be, Float32))
            px = :(_mma_zero(be, Float32))
            for d in 1:ND
                sx = :(_mma(be, Qf[$r][$d], $(kf[d]), $sx))
                px = :(_mma(be, Df[$r][$d], $(vf[d]), $px))
            end
            push!(body, quote
                s1, s2 = _mma_pair(be, $sx)
                dp1, dp2 = _mma_pair(be, $px)
                c = Int32($(8kb)) + col
                live1 = @inbounds(ms[c+Int32(1)]) == 0f0
                live2 = @inbounds(ms[c+Int32(2)]) == 0f0
                b1, b2 = _mma_bias2(bias, bbs[$r], t + c, n, Val(HB))
                l2, dlt = rs[$r]
                p1 = live1 ? exp2(s1 + b1 - l2) : 0f0
                p2 = live2 ? exp2(s2 + b2 - l2) : 0f0
                ds1 = live1 ? p1 * (dp1 - dlt) : 0f0
                ds2 = live2 ? p2 * (dp2 - dlt) : 0f0
                $(Symbol(:sf_, r)) = _mma_frag(be, eltype(kn), ds1, ds2)
            end)
        end
        for d in 1:ND
            push!(body, :(kn_ = _mma_load(be, kn, $(8kb + 1), $(8(d - 1) + 1))))
            for r in 1:R
                push!(body, :($(dq[r, d]) = _mma(be, $(Symbol(:sf_, r)), kn_, $(dq[r, d]))))
            end
        end
    end
    return Expr(:block, Expr(:meta, :inline), body...,
        Expr(:tuple, (Expr(:tuple, (dq[r, d] for d in 1:ND)...) for r in 1:R)...))
end

@inline _mma_row_stats(lse, delta, lrow, j0, n, ::Val{R}) where {R} = ntuple(Val(R)) do r
    j = min(j0 + Int32(8 * (r - 1)), n - Int32(1))
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
    BQ = 8 * R * NSG
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
    qt = g % nqt
    rest = g ÷ nqt
    i = rest % nouter
    bh = rest ÷ nouter
    hi = bh % nh
    bi = bh ÷ nh
    row = (bi * nouter + i) * n
    step = nh * Int32(DIM)
    base = row * step + hi * Int32(DIM)
    q0 = qt * Int32(BQ)
    r0 = sg * Int32(8 * R)
    fr, col = _mma_lane(be, lane)
    j0 = q0 + r0 + fr
    lrow = ((bi * nouter + i) * nh + hi) * n
    mb = ((bh ÷ mheads) * nouter + i) * n

    e = tid
    while e < Int32(BQ * DIM)
        d = e % Int32(DIM); r = e ÷ Int32(DIM); j = q0 + r
        x = j < n ? Float32(unsafe_load(q, base + j * step + d + Int32(1))) * scale * _LOG2E : 0f0
        @inbounds qs[r+Int32(1), d+Int32(1)] = T(x)
        e += NT
    end
    @synchronize
    Qf = _mma_rows(be, qs, r0, Val(R), Val(DIM ÷ 8))
    @synchronize
    e = tid
    while e < Int32(BQ * DIM)
        d = e % Int32(DIM); r = e ÷ Int32(DIM); j = q0 + r
        @inbounds qs[r+Int32(1), d+Int32(1)] = j < n ? unsafe_load(dO, base + j * step + d + Int32(1)) : zero(T)
        e += NT
    end
    @synchronize
    Df = _mma_rows(be, qs, r0, Val(R), Val(DIM ÷ 8))
    dQ = _mma_zeros(be, Val(R), Val(DIM ÷ 8))
    rs = _mma_row_stats(lse, delta, lrow, j0, n, Val(R))
    bbs = _mma_bias_rows(bh, n, j0, Val(R))

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
            Val(DIM ÷ 8), Val(BK ÷ 8), Val(HB))
        t += Int32(BK)
    end
    _mma_store_rows!(be, dq, dQ, scale, base, step, j0, n, col, Val(R), Val(DIM ÷ 8), T)
end
