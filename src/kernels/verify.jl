# The correctness gate the tuner applies to every candidate configuration before its timing counts.
#
# A tuner that only times candidates trusts every compiled configuration to be correct. That trust
# has been broken before in exactly the ways a wider search makes more likely: over-dispatching a
# pipeline past its occupancy limit returns garbage AND an impossibly fast time (see
# `src/kernels/tune.jl`), and a code generator can miscompile one tile shape while getting its
# neighbours right. The more knobs the search turns, the more binaries are produced that no test ever
# ran. So each one is checked against the plain-loop reference below, on a small problem chosen to
# reach every addressing path, before it is allowed to win.
#
# The reference speaks the kernels' own packed layout (`[dim, n, nouter, bh]`, `bias` keys-fastest
# `[n, n, bh]`, `mask` a DROP mask `[n, nouter, bh ÷ mheads]`) and computes in `Float64`. It shares no
# arithmetic with the kernels: no tiling, no online softmax, no base 2.

"""
    reference_forward(q, k, v, bias, mask, mheads) -> (o, lse)

Plain-loop attention in `Float64`, packed layout. A row with no live key attends uniformly: `o` is the
mean of `v` over the `n` keys and `lse = -Inf` (the marker the backward reads), which is what the kernel
is specified to write for it, and what a finite `-1e9` mask gives in OpenFold and Boltz2.
"""
function reference_forward(q, k, v, bias, mask, mheads)
    dim, n, nouter, bh = size(q)
    scale = inv(sqrt(dim))
    o = zeros(Float64, dim, n, nouter, bh)
    lse = zeros(Float64, n, nouter, bh)
    s = zeros(Float64, n)
    for b in 1:bh, i in 1:nouter, j in 1:n
        mrow = view(mask, :, i, (b - 1) ÷ mheads + 1)
        for kk in 1:n
            acc = 0.0
            for d in 1:dim
                acc += q[d, j, i, b] * k[d, kk, i, b]
            end
            s[kk] = mrow[kk] ? -Inf : acc * scale + bias[kk, j, b]
        end
        m = maximum(s)
        if m == -Inf
            for d in 1:dim
                o[d, j, i, b] = sum(kk -> v[d, kk, i, b], 1:n) / n
            end
            lse[j, i, b] = -Inf
            continue
        end
        l = sum(x -> exp(x - m), s)
        lse[j, i, b] = m + log(l)
        for d in 1:dim
            acc = 0.0
            for kk in 1:n
                acc += exp(s[kk] - m) * v[d, kk, i, b]
            end
            o[d, j, i, b] = acc / l
        end
    end
    return o, lse
end

"""
    reference_backward(q, k, v, bias, mask, mheads, dO) -> (dq, dk, dv, dbias)

The analytic gradient of [`reference_forward`](@ref), `Float64`, packed layout. `dbias` is keys-fastest
`[n, n, bh]` and summed over the outer axis, as `bias` has none.
"""
function reference_backward(q, k, v, bias, mask, mheads, dO)
    dim, n, nouter, bh = size(q)
    scale = inv(sqrt(dim))
    o, lse = reference_forward(q, k, v, bias, mask, mheads)
    dq, dk, dv = zeros(Float64, size(q)), zeros(Float64, size(k)), zeros(Float64, size(v))
    dbias = zeros(Float64, n, n, bh)
    for b in 1:bh, i in 1:nouter, j in 1:n
        if !isfinite(lse[j, i, b])      # no live key: uniform attention, `dV` only (the logits were replaced)
            for kk in 1:n, d in 1:dim
                dv[d, kk, i, b] += dO[d, j, i, b] / n
            end
            continue
        end
        mrow = view(mask, :, i, (b - 1) ÷ mheads + 1)
        delta = sum(o[d, j, i, b] * dO[d, j, i, b] for d in 1:dim)
        for kk in 1:n
            mrow[kk] && continue
            acc = 0.0
            for d in 1:dim
                acc += q[d, j, i, b] * k[d, kk, i, b]
            end
            p = exp(acc * scale + bias[kk, j, b] - lse[j, i, b])
            dp = sum(dO[d, j, i, b] * v[d, kk, i, b] for d in 1:dim)
            ds = p * (dp - delta)
            dbias[kk, j, b] += ds
            for d in 1:dim
                dq[d, j, i, b] += ds * scale * k[d, kk, i, b]
                dk[d, kk, i, b] += ds * scale * q[d, j, i, b]
                dv[d, kk, i, b] += p * dO[d, j, i, b]
            end
        end
    end
    return dq, dk, dv, dbias
end

"""
    verification_problem(dim, T; seed=0) -> NamedTuple

The fixed problem every candidate is checked on, with its reference results.

Small enough to cost milliseconds, shaped to reach every addressing path a configuration can get
wrong: `n = 67` is prime, so it is ragged against every tile and group size; `nouter = 5 ≠ n`
exercises the rectangular strides; `bh = 4` with `mheads = 2` gives two batch elements with
DIFFERENT masks, so a wrong batch stride on the mask is visible; the mask drops ~15% of keys and
every row keeps at least one live key, except outer slice 2 of the FIRST batch mask, which drops every key
(the uniform-attention path) while the second batch mask keeps it live. Operands are generated in `Float64` and rounded to `T` once, and the
reference is computed FROM the rounded values, so storage rounding is not counted as kernel error.

`o`, `lse` and `delta` are provided as inputs too: each backward stage is checked on its own against
exact residuals, rather than inheriting whatever error the forward candidate happened to make.
"""
function verification_problem(dim::Integer, ::Type{T}; seed::Integer=0) where {T}
    n, nouter, bh, mheads = 67, 5, 4, 2
    rng = Random.Xoshiro(seed)
    r(dims...) = T.(randn(rng, dims...))
    q, k, v, dO = r(dim, n, nouter, bh), r(dim, n, nouter, bh), r(dim, n, nouter, bh), r(dim, n, nouter, bh)
    bias = r(n, n, bh)
    mask = rand(rng, n, nouter, bh ÷ mheads) .< 0.15
    for b in axes(mask, 3), i in axes(mask, 2)
        mask[rand(rng, 1:n), i, b] = false   # at least one live key per row
    end
    mask[:, 2, 1] .= true                    # one fully masked slice, only under the first batch mask
    f64(x) = Float64.(x)
    o, lse = reference_forward(f64(q), f64(k), f64(v), f64(bias), mask, mheads)
    dq, dk, dv, dbias = reference_backward(f64(q), f64(k), f64(v), f64(bias), mask, mheads, f64(dO))
    delta = dropdims(sum(o .* f64(dO); dims=1); dims=1)
    return (; n, nouter, bh, mheads, dim=Int(dim), q, k, v, bias, mask, dO,
        o=T.(o), lse=Float32.(lse), delta=Float32.(delta),
        ref=(; o, lse, dq, dk, dv, dbias))
end

"""
    verification_tolerance(T) -> Float64

Largest accepted `max|device - reference| / max|reference|` per output. Measured on the
verification problem (`:default`, dim 16/32): `Float32` ~5e-7, `Float16` 2–4e-4, `BFloat16` 2–4e-3 —
the rounding of the outputs to the storage type. The bounds leave margin for reassociation and for
16-bit matrix operands (`:flash_mma`), and still catch any addressing or tiling error, which produces
errors of order one.
"""
verification_tolerance(::Type{Float32}) = 1e-4
verification_tolerance(::Type{Float16}) = 1e-2
verification_tolerance(::Type{BFloat16}) = 2e-2
verification_tolerance(::Type) = 1e-2

"""
    verification_error(device, reference) -> Float64

`max|device - reference| / max|reference|` over the entries where the reference is finite, in `Float64`.
`Inf` for a non-finite device value there, so a NaN can never pass by comparing false against the bound.
A reference entry that is itself infinite (the `lse = -Inf` marker of a fully masked row) must be
reproduced EXACTLY, and counts as an error otherwise.
"""
function verification_error(device, reference)
    d = Float64.(device)
    marker = .!isfinite.(reference)
    d[marker] == reference[marker] || return Inf
    fin = .!marker
    all(isfinite, d[fin]) || return Inf
    any(fin) || return 0.0
    r = reference[fin]
    return maximum(abs.(d[fin] .- r)) / max(maximum(abs.(r)), eps())
end
