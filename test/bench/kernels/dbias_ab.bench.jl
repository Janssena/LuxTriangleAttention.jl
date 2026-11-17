# A/B: the SHIPPED backward (atomic `dbias` inside `flash_bwd_dq!`, `dbias` zero-filled first) vs the
# SPLIT alternative (a `dq` kernel without `dbias` + a separate non-atomic `dbias` kernel, one thread
# per (key, query, bh) reducing over the outer index). `dk`/`dv` binaries are shared.
#
# The split kernels are NOT part of the package — they were tried as a fix for the traced-`dbias`
# bug and rejected on runtime (docs/done/BUG_traced_dbias.md, §3). They live here so that comparison
# stays reproducible, e.g. for the long-outer-extent hybrid (Option 3 in that doc).
# Direct-Metal launches on packed `[dim, n, nouter, bh]` buffers (no XLA), one process, A/B runs
# interleaved to spread thermal drift. Columns labelled "old" are the shipped kernel, "new" the split.
#
#     julia --project=test test/bench/kernels/dbias_ab.bench.jl
#
# Run it on an otherwise idle GPU: the largest case allocates ~3 GB of device memory.

using Printf, Statistics, Random
using LuxTriangleAttention, Metal
const LTA = LuxTriangleAttention
import KernelDispatch

module Split
using LuxTriangleAttention.KernelAbstractions
import LuxTriangleAttention: _kload, _kdot, _kaccum, _kstore!

# `flash_bwd_dq!` without the `dbias` output and its atomic add.
@kernel unsafe_indices = true function flash_bwd_dq_split!(dq, q, k, v, bias, mask, dO, lse, delta,
    scale, n, nouter, nh, mheads, ::Val{T}, ::Val{DIM}, ::Val{TG}, ::Val{KT}) where {T,DIM,TG,KT}
    ks = @localmem T (DIM, KT)
    vs = @localmem T (DIM, KT)

    gx, gy, gz = @index(Group, NTuple)
    tid = Int32(@index(Local, Linear))

    jt = Int32(gx) - Int32(1)
    i = Int32(gy) - Int32(1)
    bh = Int32(gz) - Int32(1)
    j = jt * Int32(TG) + tid - Int32(1)

    active = j < n
    js = active ? j : Int32(0)

    hi = bh % nh
    bi = bh ÷ nh
    nn = n * n
    row = (bi * nouter + i) * n
    hstep = nh * Int32(DIM)
    qb = (row + js) * hstep + hi * Int32(DIM)
    kb = row * hstep + hi * Int32(DIM)
    bb = bh * nn + js * n
    mb = ((bh ÷ mheads) * nouter + i) * n
    lb = ((bi * nouter + i) * nh + hi) * n + j + Int32(1)

    qv = _kload(q, qb, Val(DIM))
    dov = _kload(dO, qb, Val(DIM))
    lse_j = active ? unsafe_load(lse, lb) : zero(Float32)
    delta_j = active ? unsafe_load(delta, lb) : zero(Float32)

    dqv = ntuple(Returns(zero(Float32)), Val(DIM))

    t = Int32(0)
    while t < n
        @synchronize
        e = tid
        while e <= Int32(KT) * Int32(DIM)
            d0 = (e - Int32(1)) % Int32(DIM)
            c0 = (e - Int32(1)) ÷ Int32(DIM)
            kk = t + c0
            vk = kk < n ? unsafe_load(k, kb + kk * hstep + d0 + Int32(1)) : zero(T)
            vv = kk < n ? unsafe_load(v, kb + kk * hstep + d0 + Int32(1)) : zero(T)
            @inbounds ks[d0+Int32(1), c0+Int32(1)] = vk
            @inbounds vs[d0+Int32(1), c0+Int32(1)] = vv
            e += Int32(TG)
        end
        @synchronize

        if active
            for c in Int32(1):Int32(KT)
                kk = t + c - Int32(1)
                if kk < n
                    @inbounds masked = unsafe_load(mask, mb + kk + Int32(1))
                    if !masked
                        s = _kdot(qv, ks, c) * scale +
                            Float32(unsafe_load(bias, bb + kk + Int32(1)))
                        p = exp(s - lse_j)
                        dp = _kdot(dov, vs, c)
                        ds = p * (dp - delta_j)
                        dqv = _kaccum(dqv, ds * scale, ks, c)
                    end
                end
            end
        end
        t += Int32(KT)
    end
    active && _kstore!(dq, dqv, one(Float32), qb, T)
end

# One thread per (key, query, bh), reducing over the outer index in-thread: every `dbias` element is
# written once by a plain store. Grid: `cld(n, TG) * n * bh` groups, key tile fastest, decoded from
# the LINEAR group index (the KA context passed in carries the forward's iterspace and is unused).
@kernel unsafe_indices = true function flash_bwd_dbias!(dbias, q, k, v, bias, mask, dO, lse, delta,
    scale, n, nouter, nh, mheads, ::Val{T}, ::Val{DIM}, ::Val{TG}, ::Val{KT}) where {T,DIM,TG,KT}
    qs = @localmem T (DIM, KT)
    dos = @localmem T (DIM, KT)
    lses = @localmem Float32 (KT,)
    deltas = @localmem Float32 (KT,)

    g = Int32(@index(Group, Linear)) - Int32(1)
    tid = Int32(@index(Local, Linear))

    ntiles = (n + Int32(TG) - Int32(1)) ÷ Int32(TG)
    ktile = g % ntiles
    rest = g ÷ ntiles
    js = rest % n
    bh = rest ÷ n
    kk = ktile * Int32(TG) + tid - Int32(1)

    active = kk < n
    kks = active ? kk : Int32(0)

    hi = bh % nh
    bi = bh ÷ nh
    hstep = nh * Int32(DIM)
    bb = bh * (n * n) + js * n + kks + Int32(1)
    mb0 = (bh ÷ mheads) * nouter
    bval = Float32(unsafe_load(bias, bb))
    acc = zero(Float32)

    t = Int32(0)
    while t < nouter
        @synchronize
        e = tid
        while e <= Int32(KT) * Int32(DIM)
            d0 = (e - Int32(1)) % Int32(DIM)
            c0 = (e - Int32(1)) ÷ Int32(DIM)
            ii = t + c0
            iis = ii < nouter ? ii : Int32(0)
            qo = ((bi * nouter + iis) * n + js) * hstep + hi * Int32(DIM) + d0 + Int32(1)
            @inbounds qs[d0+Int32(1), c0+Int32(1)] = ii < nouter ? unsafe_load(q, qo) : zero(T)
            @inbounds dos[d0+Int32(1), c0+Int32(1)] = ii < nouter ? unsafe_load(dO, qo) : zero(T)
            if d0 == Int32(0)
                lo = ((bi * nouter + iis) * nh + hi) * n + js + Int32(1)
                @inbounds lses[c0+Int32(1)] = ii < nouter ? unsafe_load(lse, lo) : zero(Float32)
                @inbounds deltas[c0+Int32(1)] = ii < nouter ? unsafe_load(delta, lo) : zero(Float32)
            end
            e += Int32(TG)
        end
        @synchronize

        if active
            for c in Int32(1):Int32(KT)
                ii = t + c - Int32(1)
                if ii < nouter
                    masked = unsafe_load(mask, (mb0 + ii) * n + kks + Int32(1))
                    if !masked
                        kb = ((bi * nouter + ii) * n + kks) * hstep + hi * Int32(DIM)
                        s = _kdot(_kload(k, kb, Val(DIM)), qs, c) * scale + bval
                        p = exp(s - lses[c])
                        dp = _kdot(_kload(v, kb, Val(DIM)), dos, c)
                        acc += p * (dp - deltas[c])
                    end
                end
            end
        end
        t += Int32(KT)
    end

    active && unsafe_store!(dbias, acc, bb)
end

# The split kernels' arguments after KA's context, for `KernelDispatch.ka_compile`.
A(x) = AbstractArray{x}
argtypes_dq(T, dim, tg, kt) = (A(T), A(T), A(T), A(T), A(T), A(Bool), A(T), A(Float32), A(Float32),
    Float32, Int32, Int32, Int32, Int32, Val{T}, Val{dim}, Val{tg}, Val{kt})
argtypes_dbias(T, dim, tg, kt) = (A(Float32), A(T), A(T), A(T), A(T), A(Bool), A(T), A(Float32),
    A(Float32), Float32, Int32, Int32, Int32, Int32, Val{T}, Val{dim}, Val{tg}, Val{kt})
end

const _SPLIT = Dict{Tuple,Any}()
const _SPLIT_DIR = mktempdir()
function build_split(T, dim, tg, kt)
    get!(_SPLIT, (T, dim, tg, kt)) do
        mk(nm, argtypes, kern) = KernelDispatch.ka_compile(KernelDispatch.MetalBackendTag(), kern,
            argtypes(T, dim, tg, kt); tg, name="split_$(nm)_$(lowercase(string(nameof(T))))_d$(dim)_t$(tg)_k$(kt)", dir=_SPLIT_DIR)
        (dq=mk("dq", Split.argtypes_dq, Split.flash_bwd_dq_split!),
            dbias=mk("dbias", Split.argtypes_dbias, Split.flash_bwd_dbias!))
    end
end

maxtg(b) = KernelDispatch.max_threads(KernelDispatch.MetalBackendTag(), b)

devbytes() = Int(Metal.device().currentAllocatedSize)

med(f; reps) = (f(); f(); median([@elapsed(f()) for _ in 1:reps]))

function bench_case(; dim, H, n, nouter, reps=15)
    T = Float32
    bh = H
    tref = LTA.calibrate!(T; dim, n)
    # One (tg, kt) for every kernel in this A/B, taken from the `:dq` stage's calibration.
    cfg = LTA.choose_config(tref[:dq])                 # (; tg, kt), tuned at this n
    tg, kt = cfg.tg, cfg.kt
    rng = Xoshiro(42)
    mk4() = MtlArray(randn(rng, T, dim, n, nouter, bh))
    q, k, v, dO = mk4(), mk4(), mk4(), mk4()
    bias = MtlArray(randn(rng, T, n, n, bh))
    mask = MtlArray(falses(n, nouter))
    o = MtlArray(zeros(T, dim, n, nouter, bh))
    lse = MtlArray(zeros(Float32, n, nouter, bh))
    bias_t = permutedims(bias, (2, 1, 3))
    tr = (Float32(inv(sqrt(dim))), Int32(n), Int32(nouter), Int32(1), Int32(bh))
    KA = LTA.FlashScalar()
    KernelDispatch.run_binary(LTA.build_stage(KA, :fwd, T, dim, (; tg, kt, sb=min(4, kt))),
        LTA.stage_args(KA, Val(:fwd), (; o, lse, q, k, v, bias_t, mask), tr)...; extent=(n, nouter, bh))
    delta = LTA.flash_bwd_delta(o, dO)
    dq, dk, dv = mk4(), mk4(), mk4()
    dbias_old = MtlArray(zeros(Float32, n, n, bh))
    dbias_new = MtlArray(zeros(Float32, n, n, bh))
    Metal.synchronize()

    # The shipped kernels. The shipped `dq` reads the transposed bias and writes the transposed
    # `dbias` (`:bias_t`); `dk`/`dv` and the split kernels read the bias as given.
    shipped = (dq=LTA.build_stage(KA, :dq, T, dim, cfg), dk=LTA.build_stage(KA, :dk, T, dim, cfg),
        dv=LTA.build_stage(KA, :dv, T, dim, cfg))
    new = build_split(T, dim, tg, kt)
    # The shipped stages' arguments, as the op binds them (`stage_args`).
    roles = (; q, k, v, bias, bias_t, mask, dO, lse, delta, dq, dk, dv, dbias_t=dbias_old)
    # Every kernel covers (n, nouter, bh); the split `dbias` decodes (key tile, query, bh) from its
    # linear group index, so it covers (n, n, bh).
    L(b, args...; extent=(n, nouter, bh)) = KernelDispatch.prepare_launch(b, args...; extent)
    l_dk = L(shipped.dk, LTA.stage_args(KA, Val(:dk), roles, tr)...)
    l_dv = L(shipped.dv, LTA.stage_args(KA, Val(:dv), roles, tr)...)
    l_dqold = L(shipped.dq, LTA.stage_args(KA, Val(:dq), roles, tr)...)
    l_dqnew = L(new.dq, dq, q, k, v, bias, mask, dO, lse, delta, tr...)
    l_db = L(new.dbias, dbias_new, q, k, v, bias, mask, dO, lse, delta, tr...; extent=(n, n, bh))

    dk_() = KernelDispatch.execute(l_dk)
    dv_() = KernelDispatch.execute(l_dv)
    fill_() = (fill!(dbias_old, 0f0); Metal.synchronize())
    dqold_() = KernelDispatch.execute(l_dqold)
    dqnew_() = KernelDispatch.execute(l_dqnew)
    db_() = KernelDispatch.execute(l_db)
    # GPU time of one kernel alone (no launch latency), best of `reps` timings.
    gpu(l) = minimum(KernelDispatch.time_binary(l) for _ in 1:reps)

    old_all() = (fill_(); dqold_(); dk_(); dv_())
    new_all() = (dqnew_(); dk_(); dv_(); db_())

    # correctness cross-check before timing
    old_all(); new_all()
    old_db = permutedims(Array(dbias_old), (2, 1, 3))       # back from the shipped `:bias_t` layout
    err = maximum(abs, old_db .- Array(dbias_new)) / maximum(abs, old_db)

    # device allocation growth across one full backward (temporaries only; outputs preallocated)
    GC.gc(); b0 = devbytes(); old_all(); Metal.synchronize(); mold = devbytes() - b0
    GC.gc(); b0 = devbytes(); new_all(); Metal.synchronize(); mnew = devbytes() - b0

    # interleave A/B to spread thermal drift evenly
    to, tn = Float64[], Float64[]
    for _ in 1:reps
        push!(to, @elapsed old_all()); push!(tn, @elapsed new_all())
    end
    r = (; dim, H, n, nouter, tg, kt,
        t_old=median(to), t_new=median(tn),
        t_fill=med(fill_; reps), t_dqold=gpu(l_dqold), t_dqnew=gpu(l_dqnew),
        t_db=gpu(l_db), t_dk=gpu(l_dk), t_dv=gpu(l_dv),
        mold, mnew, err,
        maxtg_dqold=maxtg(shipped.dq), maxtg_dqnew=maxtg(new.dq), maxtg_db=maxtg(new.dbias))
    return r
end

ms(x) = @sprintf("%7.2f", 1e3x)

function main()
    cases = [
        # triangle attention, Boltz2/AF2 pair shape (square)
        [(; dim=32, H=4, n=n, nouter=n) for n in (64, 128, 256, 384, 512)]...,
        # MSA row attention (bias over residues, reduction over sequences)
        (; dim=32, H=8, n=256, nouter=512),
        # extra-MSA row attention: long outer reduction, the new kernel's worst case
        (; dim=8, H=8, n=256, nouter=1024),
        (; dim=8, H=8, n=384, nouter=4096),
    ]
    println("config                         tg  kt | old total  new total  Δ%   | fill  dq_old  dq_new  dbias   dk     dv    (ms, kernels: GPU time) | Δdev old/new | rel.err | maxTG dq_old/dq_new/dbias")
    for c in cases
        r = bench_case(; c...)
        @printf("dim=%-2d H=%d n=%-4d nouter=%-5d %4d %3d | %s   %s  %+5.1f | %s %s %s %s %s %s | %d/%d | %.1e | %d/%d/%d\n",
            r.dim, r.H, r.n, r.nouter, r.tg, r.kt, ms(r.t_old), ms(r.t_new),
            100 * (r.t_new / r.t_old - 1), ms(r.t_fill), ms(r.t_dqold), ms(r.t_dqnew), ms(r.t_db),
            ms(r.t_dk), ms(r.t_dv), r.mold, r.mnew, r.err,
            r.maxtg_dqold, r.maxtg_dqnew, r.maxtg_db)
        flush(stdout)
    end
end

main()
