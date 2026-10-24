# A second flash-attention kernel, `:flash_mma`: the same algorithm as `flash/` (online softmax,
# Float32 accumulation, base-2 scores, the same residuals), with every matrix product on the GPU's
# matrix units — 8×8 `simdgroup_matrix` multiply-accumulates on Metal. Selected with
# `use_kernel!(:flash_mma)`; the original kernel (`:$(RESERVED_KERNEL_NAME)`, `flash/`) is untouched
# and stays the default. Design, measurements and the road to here: docs/done/FEATURE_flash_mma.md.
#
# ## Backend hooks
#
# Like `flash/`, this file names no backend symbol. The few operations KernelAbstractions has no
# notion of — matrix fragments and the lanes that hold them — are hooks, declared below and
# implemented per backend (Metal: `ext/LuxTriangleAttentionMetalExt/flash_mma.jl`), dispatched on
# KernelDispatch' backend tag (`MetalBackendTag`, ...), passed as a `Val{typeof(tag)}` kernel parameter
# that takes no buffer slot.
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
# The algorithm assumes the fragment shape every hook speaks: an 8×8 fragment in which each of a
# 32-lane SIMD group's lanes holds TWO ADJACENT ENTRIES OF ONE ROW, so the four lanes holding a row
# can reduce it with shuffles (`_mma_rowmax`/`_mma_rowsum`). Apple's simdgroup matrices are laid out
# this way (measured: `test/ext/metal/flash_mma.test.jl`); so is the accumulator of NVIDIA's
# `mma.sync` m16n8, row for row, which is what a CUDA port would implement the hooks with.
#
# ## Why the softmax runs in registers
#
# The scores of a key tile are never written anywhere: each lane reads its two entries of each score
# fragment, adds bias and mask, reduces the row maximum and sum across the row's four lanes, and
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
# A group covers `rows = 8·R·NSG` queries (or keys) with `32·NSG` threads, which the stage's
# binary records as its `tile = (rows, 1, 1)`, so a launch of `extent = (n, nouter, H·B)` dispatches the
# right number of groups.

# ---------------------------------------------------------------------------------------------
# Hooks (implemented per backend)
# ---------------------------------------------------------------------------------------------

"""
    _mma_load(be, a, r, c) -> fragment

The 8×8 fragment of local-memory matrix `a` whose top-left entry is `a[r, c]` (1-based). `a` is a
column-major `@localmem` array; the fragment is the ordinary (non-transposed) matrix block.
"""
function _mma_load end

"""`_mma_zero(be, E)`: a fragment of zeros of element type `E`, valid as an accumulator."""
function _mma_zero end

"""`_mma(be, a, b, c) -> c + a·b`, Float32 accumulator `c`; `a`, `b` both Float32, Float16 or BFloat16."""
function _mma end

"""`_mma_pair(be, f) -> (x, y)`: this lane's two entries of fragment `f`, as Float32."""
function _mma_pair end

"""`_mma_setpair(be, f, x, y) -> fragment`: `f` with this lane's two entries replaced."""
function _mma_setpair end

"""`_mma_lane(be, lane) -> (row, col)`: the (0-based) entries `(row, col)`, `(row, col+1)` that
lane `lane` (0-based) holds in every fragment."""
function _mma_lane end

"""`_mma_rowmax(be, x)`, `_mma_rowsum(be, x)`: reduce `x` across the lanes holding one row."""
function _mma_rowmax end
function _mma_rowsum end

@inline _mma_frag(be, ::Type{E}, x, y) where {E} = _mma_setpair(be, _mma_zero(be, E), x, y)

# ---------------------------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------------------------

# `R` row blocks × `ND` channel blocks of fragments, loaded from a [rows, DIM] local array.
@inline _mma_rows(be, a, r0, ::Val{R}, ::Val{ND}) where {R,ND} =
    ntuple(rb -> ntuple(d -> _mma_load(be, a, r0 + 8(rb - 1) + 1, 8(d - 1) + 1), Val(ND)), Val(R))

@inline _mma_zeros(be, ::Val{R}, ::Val{ND}) where {R,ND} =
    ntuple(_ -> ntuple(_ -> _mma_zero(be, Float32), Val(ND)), Val(R))

# Store each lane's two entries of `F[rb][d]` (times `w`) straight to device memory: the row is the
# lane's row (`rows` for block 1), the channels `8(d-1) + col` and the next.
@generated function _mma_store_rows!(be, out, F, w, base, step, rows, n, col, ::Val{R}, ::Val{ND},
    ::Type{T}) where {R,ND,T}
    body = Expr[]
    for rb in 1:R
        stores = map(1:ND) do d
            quote
                x, y = _mma_pair(be, F[$rb][$d])
                unsafe_store!(out, T(x * w), ob + Int32($(8(d - 1))))
                unsafe_store!(out, T(y * w), ob + Int32($(8(d - 1) + 1)))
            end
        end
        push!(body, quote
            rr = rows + Int32($(8(rb - 1)))
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

# One key tile for the R query blocks of this simdgroup. `st[rb] = (m, l)` of this lane's row.
@generated function _mma_fwd_tile(be, O, st, Qf, ks, vs, ms, bias, bbs, t, n, col, ::Val{R},
    ::Val{ND}, ::Val{NKB}, ::Val{HB}) where {R,ND,NKB,HB}
    body = Expr[]
    kf(kb, d) = Symbol(:kf_, kb, :_, d)
    for kb in 0:NKB-1, d in 1:ND
        push!(body, :($(kf(kb, d)) = _mma_load(be, ks, $(8(d - 1) + 1), $(8kb + 1))))
    end
    newst = Any[]
    for rb in 1:R
        s = [(Symbol(:s1_, rb, :_, kb), Symbol(:s2_, rb, :_, kb)) for kb in 0:NKB-1]
        for kb in 0:NKB-1
            ex = :(_mma_zero(be, Float32))
            for d in 1:ND
                ex = :(_mma(be, Qf[$rb][$d], $(kf(kb, d)), $ex))
            end
            a, b = s[kb+1]
            push!(body, quote
                $a, $b = _mma_pair(be, $ex)
                c = Int32($(8kb)) + col
                b1, b2 = _mma_bias2(bias, bbs[$rb], t + c, n, Val(HB))
                $a += b1 + @inbounds(ms[c+Int32(1)])
                $b += b2 + @inbounds(ms[c+Int32(2)])
            end)
        end
        m, l, corr = Symbol(:m_, rb), Symbol(:l_, rb), Symbol(:corr_, rb)
        mx = foldl((x, y) -> :(max($x, $y)), [:(max($a, $b)) for (a, b) in s])
        push!(body, quote
            $m = max(st[$rb][1], _mma_rowmax(be, $mx))
            $corr = exp2(st[$rb][1] - $m)
        end)
        for (a, b) in s
            push!(body, :($a = exp2($a - $m)), :($b = exp2($b - $m)))
        end
        sm = foldl((x, y) -> :($x + $y), [:($a + $b) for (a, b) in s])
        push!(body, :($l = st[$rb][2] * $corr + _mma_rowsum(be, $sm)))
        for kb in 0:NKB-1
            a, b = s[kb+1]
            push!(body, :($(Symbol(:pf_, rb, :_, kb)) = _mma_frag(be, eltype(vs), $a, $b)))
        end
        for d in 1:ND
            push!(body, :($(Symbol(:o_, rb, :_, d)) = let f = O[$rb][$d]
                x, y = _mma_pair(be, f)
                _mma_setpair(be, f, x * $corr, y * $corr)
            end))
        end
        push!(newst, :(($m, $l)))
    end
    for kb in 0:NKB-1, d in 1:ND
        push!(body, :(vf = _mma_load(be, vs, $(8kb + 1), $(8(d - 1) + 1))))
        for rb in 1:R
            od = Symbol(:o_, rb, :_, d)
            push!(body, :($od = _mma(be, $(Symbol(:pf_, rb, :_, kb)), vf, $od)))
        end
    end
    Ot = Expr(:tuple, (Expr(:tuple, (Symbol(:o_, rb, :_, d) for d in 1:ND)...) for rb in 1:R)...)
    return Expr(:block, Expr(:meta, :inline), body..., :(($Ot, $(Expr(:tuple, newst...)))))
end

# Bias row offsets for this lane's query in each row block: bias[key, query j, bh] at (bh·n + j)·n.
@inline _mma_bias_rows(bh, n, j0, ::Val{R}) where {R} =
    ntuple(rb -> (bh * n + min(j0 + Int32(8 * (rb - 1)), n - Int32(1))) * n, Val(R))

@generated function _mma_fwd_store!(be, o, lse, O, st, j0, n, base, step, lrow, col, ::Val{R},
    ::Val{ND}, ::Type{T}) where {R,ND,T}
    body = Expr[]
    for rb in 1:R
        stores = map(1:ND) do d
            quote
                x, y = _mma_pair(be, O[$rb][$d])
                unsafe_store!(o, T(x * w), ob + Int32($(8(d - 1))))
                unsafe_store!(o, T(y * w), ob + Int32($(8(d - 1) + 1)))
            end
        end
        push!(body, quote
            j = j0 + Int32($(8(rb - 1)))
            if j < n
                m, l = st[$rb]
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

A group of `32·NSG` threads covers `8·R·NSG` queries; each simdgroup owns `R` blocks of 8 query rows
and holds their `Q` fragments for the whole call. Per key tile of `BK` keys the group stages `Kᵀ`,
`V` (in `PE`) and the additive key mask; see this file's header for the in-register softmax.
`Q·Kᵀ` multiplies storage-type fragments; `P·V` multiplies `PE` operands; every accumulator is
`Float32` (see "Precision" in this file's header).
"""
@kernel unsafe_indices = true function flash_mma_fwd!(o, lse, q, k, v, bias, mask,
    scale, n, nouter, nh, mheads, ::Val{BE}, ::Val{T}, ::Val{DIM}, ::Val{NSG},
    ::Val{R}, ::Val{BK}, ::Val{HB}, ::Val{HM}, ::Val{PE}) where {BE,T,DIM,NSG,R,BK,HB,HM,PE}
    be = BE()
    BQ = 8 * R * NSG
    qs = @localmem T (BQ, DIM)
    ks = @localmem T (DIM, BK)
    vs = @localmem PE (BK, DIM)
    ms = @localmem Float32 (BK,)

    NT = Int32(32 * NSG)
    g = Int32(@index(Group, Linear)) - Int32(1)
    tid = Int32(@index(Local, Linear)) - Int32(1)
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
    j0 = q0 + r0 + fr                          # this lane's query in row block 1
    mb = ((bh ÷ mheads) * nouter + i) * n

    e = tid
    while e < Int32(BQ * DIM)
        d = e % Int32(DIM); r = e ÷ Int32(DIM); j = q0 + r
        x = j < n ? Float32(unsafe_load(q, base + j * step + d + Int32(1))) : 0f0
        @inbounds qs[r+Int32(1), d+Int32(1)] = T(x * scale * _LOG2E)
        e += NT
    end
    @synchronize
    Qf = _mma_rows(be, qs, r0, Val(R), Val(DIM ÷ 8))
    O = _mma_zeros(be, Val(R), Val(DIM ÷ 8))
    st = ntuple(_ -> (_MMA_NEG, 0f0), Val(R))
    bbs = _mma_bias_rows(bh, n, j0, Val(R))

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
        O, st = _mma_fwd_tile(be, O, st, Qf, ks, vs, ms, bias, bbs, t, n, col, Val(R),
            Val(DIM ÷ 8), Val(BK ÷ 8), Val(HB))
        t += Int32(BK)
    end
    _mma_fwd_store!(be, o, lse, O, st, j0, n, base, step, (bi * nouter + i) * nh + hi, col,
        Val(R), Val(DIM ÷ 8), T)
end
