# A second flash-attention kernel, `:flash_mma`: the same algorithm as `flash/` (online softmax,
# Float32 accumulation, base-2 scores, the same residuals), with every matrix product on the GPU's
# matrix units — 8×8 `simdgroup_matrix` multiply-accumulates on Metal, `mma.sync` tensor-core products
# on NVIDIA. Selected with `use_kernel!(:flash_mma)`; the original kernel (`:$(RESERVED_KERNEL_NAME)`,
# `flash/`) is untouched and stays the default. Design, measurements and the road to here:
# docs/done/FEATURE_flash_mma.md.
#
# ## Backend hooks
#
# Like `flash/`, this file names no backend symbol. The few operations KernelAbstractions has no
# notion of — matrix fragments and the lanes that hold them — are hooks, declared below and
# implemented per backend (Metal: `ext/LuxTriangleAttentionMetalExt/flash_mma.jl`, CUDA:
# `ext/LuxTriangleAttentionCUDAExt/flash_mma.jl`), dispatched on KernelDispatch' backend tag
# (`MetalBackendTag`, `CUDABackendTag`, ...), passed as a `Val{typeof(tag)}` kernel parameter that takes no
# buffer slot.
#
# ## Precision
#
# Storage (`q`, `k`, `v`, outputs, gradients) is the caller's type `T`; `Q·Kᵀ`, `dO·Vᵀ` multiply
# storage-type fragments. Every accumulator is `Float32`, and so is the softmax state.
#
# The second product of each step (`P·V`, `P·dO`, `dS·Q`, `dS·K`) multiplies operands of type `PE`
# (a compile-time parameter): the probabilities `P` / score gradients `dS` rounded to `PE`, against
# the other operand staged in `PE`. `PE` is `Float32` or, for 16-bit storage, the storage type — the
# FlashAttention-2 convention, several times faster on matrix units with fast 16-bit inputs (NVIDIA
# tensor cores); on an M1 Pro ~3–4%, at a backward relL2 of 4.1e-4 against 2.9e-4 (Float16). Which one
# a device runs is TUNED (`tune_grid`), every candidate verified (also on a large `dO`, `DO_SCALE`).
# BFloat16 has Float32's range; a Float16 `dS` can overflow for peaked attention with a large loss-scaled
# `dO`, the FlashAttention-2 caveat. On the M1, BFloat16 operands are rejected by verification: Apple's
# compiler cannot build a bfloat fragment from registers (see `ext/LuxTriangleAttentionMetalExt/flash_mma.jl`).
#
# ## The fragment shape
#
# The algorithm is written against a fragment shape the backend states (`_mma_shape`): a product
# `C[FM×FN] += A[FM×FK]·B[FK×FN]`. Each of a 32-lane SIMD group's lanes holds, in an accumulator, TWO ADJACENT
# ENTRIES of each of `FM÷8` rows (its row SLOTS: rows `fr`, `fr+8`, …), so the lanes holding a row can reduce it
# with shuffles (`_mma_rowmax`/`_mma_rowsum`).
#
#   Metal   8×8×8,   one slot: an 8×8 `simdgroup_matrix` (measured: `test/ext/metal/flash_mma.test.jl`).
#   NVIDIA  16×8×16, two slots: `mma.sync` m16n8k16, whose accumulator holds rows g and g+8, columns 2t, 2t+1
#           (g = lane÷4, t = lane%4). Two adjacent n8 accumulators are exactly one k16 operand `A`
#           (`_mma_pfrag`), so the probabilities never leave registers there either.
#
# Everything the kernels index by "row" is indexed by SLOT `u = 1…R·FM÷8`: slot `u` is row `fr + 8(u-1)` of the
# simdgroup's block, whatever the shape, and fragment `rb` provides slots `NR(rb-1)+1 … NR·rb`.
#
# ## Why the softmax runs in registers
#
# The scores of a key tile are never written anywhere: each lane reads its entries of each score
# fragment, adds bias and mask, reduces the row maximum and sum across the row's lanes, and
# writes the probabilities into a fresh fragment for the `P·V` product. The accumulator is rescaled
# entry by entry the same way. A first version that staged scores and probabilities through local
# memory and rescaled with a diagonal matrix product was no faster than the scalar kernel; this one
# is 1.5–1.9× faster (forward) and 1.9–3.7× (backward). See the design doc.
#
# ## Stages
#
#   :fwd   one threadgroup per (query tile, outer index, head·batch); o, lse
#   :dkv   one threadgroup per (KEY tile, outer index, head·batch), rows = keys, everything
#          transposed: dK, dV, and dbias (atomics, as the bias is shared across the outer index)
#   :dq    one threadgroup per (query tile, …): dQ
#
# A group covers `rows = FM·R·NSG` queries (or keys) with `32·NSG` threads, which the stage's
# binary records as its `tile = (rows, 1, 1)`, so a launch of `extent = (n, nouter, H·B)` dispatches the
# right number of groups.

# ---------------------------------------------------------------------------------------------
# Hooks (implemented per backend)
# ---------------------------------------------------------------------------------------------

"""
    _mma_shape(be) -> (FM, FN, FK)

The product `C[FM×FN] += A[FM×FK]·B[FK×FN]` the backend's matrix unit computes; the kernels are written against it.
`FM` is a multiple of 8 (`FM÷8` row slots per lane, see "The fragment shape" in this file); `FN`, `FK` divide the
head dimension (`FK`) and the key / query tile (`FK`). Defaults to the 8×8×8 of [`MetalBackendTag`] for any backend
(the host side asks it too, with no device loaded).
"""
_mma_shape(::KernelBackend) = (8, 8, 8)

# The same, from the tag's TYPE, for the `@generated` builders (they see types).
_mma_dims(::Type{BE}) where {BE} = _mma_shape(BE())

"""
    _mma_load_a(be, a, r, c) -> fragment
    _mma_load_b(be, a, r, c) -> fragment

The `A` operand (`FM×FK` block) or the `B` operand (`FK×FN` block) of local-memory matrix `a` whose top-left entry is
`a[r, c]` (1-based). `a` is a column-major `@localmem` array holding the matrix as it enters the product (rows
first), so a `B` block is read down its columns.
"""
function _mma_load_a end
function _mma_load_b end

"""`_mma_zero(be, E)`: a fragment of zeros of element type `E`, valid as an accumulator."""
function _mma_zero end

"""`_mma(be, a, b, c) -> c + a·b`, Float32 accumulator `c`; `a`, `b` both Float32, Float16 or BFloat16."""
function _mma end

"""`_mma_pair(be, f, s) -> (x, y)`: this lane's two entries of row slot `s` of fragment `f`, as Float32."""
function _mma_pair end

"""`_mma_setpair(be, f, s, x, y) -> fragment`: `f` with this lane's two entries of row slot `s` replaced."""
function _mma_setpair end

"""`_mma_lane(be, lane) -> (row, col)`: the (0-based) entries `(row + 8(s-1), col)`, `(row + 8(s-1), col+1)`
that lane `lane` (0-based) holds in every accumulator, for each row slot `s`."""
function _mma_lane end

"""`_mma_rowmax(be, x)`, `_mma_rowsum(be, x)`: reduce `x` across the lanes holding one row."""
function _mma_rowmax end
function _mma_rowsum end

"""
    _mma_pfrag(be, E, p1, p2, …) -> operand

The next product's `A` operand, of element type `E`, built from accumulator entries already in registers: the pairs
`pi = (x, y)` of one row slot of one `FN`-wide block, listed block by block, slot by slot (`FK÷FN` blocks of `FM÷8`
slots). Falls back to a single pair set into a zero fragment, which is the case when the operand is one block of
one slot (Metal).
"""
@inline _mma_pfrag(be, ::Type{E}, p) where {E} = _mma_frag(be, E, p[1], p[2])
@inline _mma_frag(be, ::Type{E}, x, y) where {E} = _mma_setpair(be, _mma_zero(be, E), 1, x, y)

# ---------------------------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------------------------

# `R` fragments × the `A` operands along the head dimension, loaded from a [rows, DIM] local array.
@generated function _mma_rows(be, a, r0, ::Val{R}, ::Val{DIM}) where {R,DIM}
    FM, FN, FK = _mma_dims(be)
    rows = (Expr(:tuple, (:(_mma_load_a(be, a, r0 + $(FM * (rb - 1) + 1), $(FK * (k - 1) + 1))) for k in 1:DIM÷FK)...)
            for rb in 1:R)
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, rows...))
end

# `R` fragments × the accumulators along the head dimension (`FN` channels each), zeroed.
@generated function _mma_zeros(be, ::Val{R}, ::Val{DIM}) where {R,DIM}
    FM, FN, FK = _mma_dims(be)
    rows = (Expr(:tuple, (:(_mma_zero(be, Float32)) for _ in 1:DIM÷FN)...) for _ in 1:R)
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, rows...))
end

# Store each lane's entries of `F[rb][d]` (times `w`) straight to device memory: the row is the lane's row of the
# slot (`rows` + 8(u-1)), the channels `FN(d-1) + col` and the next.
@generated function _mma_store_rows!(be, out, F, w, base, step, rows, n, col, ::Val{R}, ::Val{DIM},
    ::Type{T}) where {R,DIM,T}
    FM, FN, FK = _mma_dims(be)
    NR = FM ÷ 8
    body = Expr[]
    for rb in 1:R, s in 1:NR
        u = NR * (rb - 1) + s
        stores = map(1:DIM÷FN) do d
            quote
                x, y = _mma_pair(be, F[$rb][$d], $s)
                unsafe_store!(out, T(x * w), ob + Int32($(FN * (d - 1))))
                unsafe_store!(out, T(y * w), ob + Int32($(FN * (d - 1) + 1)))
            end
        end
        push!(body, quote
            rr = rows + Int32($(8 * (u - 1)))
            if rr < n
                ob = base + rr * step + col + Int32(1)
                $(stores...)
            end
        end)
    end
    return Expr(:block, Expr(:meta, :inline), body..., :(return nothing))
end

# Bias (keys fastest) for this lane's two keys `kc`, `kc+1`, in base 2; addresses clamped.
@inline function _mma_bias2(bias, rowoff, kc, n, ::Val{HB}) where {HB}
    HB || return 0f0, 0f0
    k1 = min(kc, n - Int32(1))
    k2 = min(kc + Int32(1), n - Int32(1))
    return Float32(unsafe_load(bias, rowoff + k1 + Int32(1))) * _LOG2E,
           Float32(unsafe_load(bias, rowoff + k2 + Int32(1))) * _LOG2E
end

# ---------------------------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------------------------

# One key tile for the R query fragments of this simdgroup. `st[u] = (m, l)` of this lane's row slot `u`.
@generated function _mma_fwd_tile(be, O, st, Qf, ks, vs, ms, bias, bbs, t, n, col, ::Val{R},
    ::Val{DIM}, ::Val{BK}, ::Val{HB}) where {R,DIM,BK,HB}
    FM, FN, FK = _mma_dims(be)
    NR, NKD, NKB, NKK, NDN, NB = FM ÷ 8, DIM ÷ FK, BK ÷ FN, BK ÷ FK, DIM ÷ FN, FK ÷ FN
    body = Expr[]
    kf(kb, k) = Symbol(:kf_, kb, :_, k)
    s1(u, kb) = Symbol(:s1_, u, :_, kb)
    s2(u, kb) = Symbol(:s2_, u, :_, kb)
    for kb in 0:NKB-1, k in 1:NKD
        push!(body, :($(kf(kb, k)) = _mma_load_b(be, ks, $(FK * (k - 1) + 1), $(FN * kb + 1))))
    end
    newst = Any[]
    for rb in 1:R
        acc(kb) = Symbol(:acc_, rb, :_, kb)
        for kb in 0:NKB-1
            ex = :(_mma_zero(be, Float32))
            for k in 1:NKD
                ex = :(_mma(be, Qf[$rb][$k], $(kf(kb, k)), $ex))
            end
            push!(body, :($(acc(kb)) = $ex))
        end
        for s in 1:NR
            u = NR * (rb - 1) + s
            for kb in 0:NKB-1
                a, b = s1(u, kb), s2(u, kb)
                push!(body, quote
                    $a, $b = _mma_pair(be, $(acc(kb)), $s)
                    c = Int32($(FN * kb)) + col
                    b1, b2 = _mma_bias2(bias, bbs[$u], t + c, n, Val(HB))
                    $a += b1 + @inbounds(ms[c+Int32(1)])
                    $b += b2 + @inbounds(ms[c+Int32(2)])
                end)
            end
            m, l, corr = Symbol(:m_, u), Symbol(:l_, u), Symbol(:corr_, u)
            mx = foldl((x, y) -> :(max($x, $y)), [:(max($(s1(u, kb)), $(s2(u, kb)))) for kb in 0:NKB-1])
            push!(body, quote
                $m = max(st[$u][1], _mma_rowmax(be, $mx))
                $corr = exp2(st[$u][1] - $m)
            end)
            for kb in 0:NKB-1
                push!(body, :($(s1(u, kb)) = exp2($(s1(u, kb)) - $m)), :($(s2(u, kb)) = exp2($(s2(u, kb)) - $m)))
            end
            sm = foldl((x, y) -> :($x + $y), [:($(s1(u, kb)) + $(s2(u, kb))) for kb in 0:NKB-1])
            push!(body, :($l = st[$u][2] * $corr + _mma_rowsum(be, $sm)))
            push!(newst, :(($m, $l)))
        end
        # The probabilities as the next product's operand, one per chunk of `FK` keys.
        for kc in 1:NKK
            ps = [:(($(s1(NR * (rb - 1) + s, NB * (kc - 1) + j)), $(s2(NR * (rb - 1) + s, NB * (kc - 1) + j))))
                  for j in 0:NB-1 for s in 1:NR]
            push!(body, :($(Symbol(:pf_, rb, :_, kc)) = _mma_pfrag(be, eltype(vs), $(ps...))))
        end
        for dn in 1:NDN
            stmts = Any[]
            for s in 1:NR
                corr = Symbol(:corr_, NR * (rb - 1) + s)
                push!(stmts, :((x, y) = _mma_pair(be, f, $s)), :(f = _mma_setpair(be, f, $s, x * $corr, y * $corr)))
            end
            push!(body, :($(Symbol(:o_, rb, :_, dn)) = $(Expr(:let, :(f = O[$rb][$dn]), Expr(:block, stmts..., :f)))))
        end
    end
    for kc in 1:NKK, dn in 1:NDN
        push!(body, :(vf = _mma_load_b(be, vs, $(FK * (kc - 1) + 1), $(FN * (dn - 1) + 1))))
        for rb in 1:R
            od = Symbol(:o_, rb, :_, dn)
            push!(body, :($od = _mma(be, $(Symbol(:pf_, rb, :_, kc)), vf, $od)))
        end
    end
    Ot = Expr(:tuple, (Expr(:tuple, (Symbol(:o_, rb, :_, d) for d in 1:NDN)...) for rb in 1:R)...)
    return Expr(:block, Expr(:meta, :inline), body..., :(($Ot, $(Expr(:tuple, newst...)))))
end

# Bias row offsets for this lane's query in each row slot: bias[key, query j, bh] at (bh·n + j)·n.
@inline _mma_bias_rows(bh, n, j0, ::Val{U}) where {U} =
    ntuple(u -> (bh * n + min(j0 + Int32(8 * (u - 1)), n - Int32(1))) * n, Val(U))

@generated function _mma_fwd_store!(be, o, lse, O, st, j0, n, base, step, lrow, col, ::Val{R},
    ::Val{DIM}, ::Type{T}) where {R,DIM,T}
    FM, FN, FK = _mma_dims(be)
    NR = FM ÷ 8
    body = Expr[]
    for rb in 1:R, s in 1:NR
        u = NR * (rb - 1) + s
        stores = map(1:DIM÷FN) do d
            quote
                x, y = _mma_pair(be, O[$rb][$d], $s)
                unsafe_store!(o, T(x * w), ob + Int32($(FN * (d - 1))))
                unsafe_store!(o, T(y * w), ob + Int32($(FN * (d - 1) + 1)))
            end
        end
        push!(body, quote
            j = j0 + Int32($(8 * (u - 1)))
            if j < n
                m, l = st[$u]
                dead = m <= 0.5f0 * _MMA_NEG     # no live key in this row: o = mean(v), lse = -Inf, as `flash/`
                w = dead ? inv(Float32(n)) : inv(l)   # (every score is `_MMA_NEG`, p = 1: O holds Σₖ vₖ)
                ob = base + j * step + col + Int32(1)
                $(stores...)
                col == Int32(0) && unsafe_store!(lse, dead ? -Inf32 : (m + log2(l)) * _LN2,
                    lrow * n + j + Int32(1))
            end
        end)
    end
    return Expr(:block, Expr(:meta, :inline), body..., :(return nothing))
end

const _MMA_NEG = -floatmax(Float32)

"""
    flash_mma_fwd!(o, lse, q, k, v, bias, mask, scale, n, nouter, nh, mheads,
        Val(BE), Val(T), Val(DIM), Val(NSG), Val(R), Val(BK), Val(HB), Val(HM), Val(PE))

Forward of `:flash_mma`. Same buffers, layout and results as [`flash_fwd!`](@ref) except that the
pair bias is read in the ordinary keys-fastest `:bias` layout (each lane reads two adjacent keys of
its own query row) rather than transposed. `mask` is the DROP mask `[n, nouter, bh ÷ mheads]`.

A group of `32·NSG` threads covers `FM·R·NSG` queries; each simdgroup owns `R` fragments of `FM` query rows
and holds their `Q` operands for the whole call. Per key tile of `BK` keys the group stages `Kᵀ`,
`V` (in `PE`) and the additive key mask; see this file's header for the in-register softmax.
`Q·Kᵀ` multiplies storage-type fragments; `P·V` multiplies `PE` operands; every accumulator is
`Float32` (see "Precision" in this file's header).
"""
@kernel unsafe_indices = true function flash_mma_fwd!(o, lse, q, k, v, bias, mask,
    scale, n, nouter, nh, mheads, ::Val{BE}, ::Val{T}, ::Val{DIM}, ::Val{NSG},
    ::Val{R}, ::Val{BK}, ::Val{HB}, ::Val{HM}, ::Val{PE}) where {BE,T,DIM,NSG,R,BK,HB,HM,PE}
    be = BE()
    FM, FN, FK = _mma_shape(be)
    U = R * (FM ÷ 8)                       # row slots this lane holds
    BQ = FM * R * NSG
    qs = @localmem T (BQ, DIM)
    ks = @localmem T (DIM, BK)
    vs = @localmem PE (BK, DIM)
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
    j0 = q0 + r0 + fr                          # this lane's query in row slot 1
    mb = (_sdiv(bh, mheads) * nouter + i) * n

    e = tid
    while e < Int32(BQ * DIM)
        d = e % Int32(DIM); r = e ÷ Int32(DIM); j = q0 + r
        x = j < n ? Float32(unsafe_load(q, base + j * step + d + Int32(1))) : 0f0
        @inbounds qs[r+Int32(1), d+Int32(1)] = T(x * scale * _LOG2E)
        e += NT
    end
    @synchronize
    Qf = _mma_rows(be, qs, r0, Val(R), Val(DIM))
    O = _mma_zeros(be, Val(R), Val(DIM))
    st = ntuple(_ -> (_MMA_NEG, 0f0), Val(U))
    bbs = _mma_bias_rows(bh, n, j0, Val(U))

    t = Int32(0)
    while t < n
        @synchronize
        e = tid
        while e < Int32(DIM * BK)
            d = e % Int32(DIM); c = e ÷ Int32(DIM); kk = t + c
            off = base + kk * step + d + Int32(1)
            @inbounds ks[d+Int32(1), c+Int32(1)] = kk < n ? unsafe_load(k, off) : zero(T)
            @inbounds vs[c+Int32(1), d+Int32(1)] = kk < n ? PE(Float32(unsafe_load(v, off))) : zero(PE)
            e += NT
        end
        e = tid
        while e < Int32(BK)
            kk = t + e
            dead = kk < n ? (HM ? unsafe_load(mask, mb + kk + Int32(1)) : false) : true
            @inbounds ms[e+Int32(1)] = dead ? _MMA_NEG : 0f0
            e += NT
        end
        @synchronize
        O, st = _mma_fwd_tile(be, O, st, Qf, ks, vs, ms, bias, bbs, t, n, col, Val(R), Val(DIM), Val(BK), Val(HB))
        t += Int32(BK)
    end
    _mma_fwd_store!(be, o, lse, O, st, j0, n, base, step, (bi * nouter + i) * nh + hi, col,
        Val(R), Val(DIM), T)
end
