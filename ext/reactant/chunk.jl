# The rolled `@trace` chunking path: `_sdpa_stable`'s traced implementation, the compile options and
# `@code_hlo` check that keep the rolled loop from being optimized away, and `_chunked_sdpa`
# itself. Everything here exists only to bound the REVERSE pass's memory (see the base package's
# `layers/chunk.jl` and `Attention`'s `chunk_size` docs for the forward-side motivation).
#
# `_sdpa_stable` and `_stopgrad` are LOCAL to this extension, not methods of a base-package function:
# nothing in `src/` ever calls either one — the rolled loop below is their only caller, and it lives
# in this same file. Giving them base-package stub methods (an earlier version of this file did)
# bought nothing, since a stub nothing calls is not an extension point; it just split one concept's
# implementation across two files for no reason.

# Enzyme's reverse of a max-reduce inside a `stablehlo.while` fails to compile ("operand #0 does not
# dominate this use"), which removed the rolled path's entire reason for existing. A softmax's max
# is a constant shift whose derivative cancels identically, so marking it non-differentiable is
# exact — and it keeps the failing pattern out of the graph.
_stopgrad(x) = Reactant.EnzymeCore.ignore_derivatives(x)

"""
    _sdpa_stable(q::AnyTracedRArray, k, v, mask, bias)

Traced scaled-dot-product attention: a SHADOW COPY of LuxLib's
`ext/ReactantExt/attention.jl::Impl.unsafe_scaled_dot_product_attention`, with one change — the
softmax's max-reduce is wrapped in `ignore_derivatives`.

Why this needs its own implementation rather than just calling `_sdpa_default` on a traced array: it
would still reach the same underlying `Lux.scaled_dot_product_attention` call, which LuxLib's own
`ReactantExt` already replaces for any traced array — but WITHOUT this fix, so the failure inside the
rolled `stablehlo.while` would be exactly the same.

Reimplemented here (not delegated to LuxLib's version) because LuxLib's traced SDPA works in
REVERSED dim order (`alpha` is `[B…, H, Nq, Nk]`, the mirror of the generic path's
`[Nk, Nq, H, …, B]`) and there is no seam to inject `ignore_derivatives` into it from outside.
Fixed here to `head_dim=1, token_dim=3, num_heads_dim=2, fdrop=identity`, which is what `Attention`
always passes, and grouped-query attention is not supported (asserted).

TEMPORARY. The real fix is that one `ignore_derivatives` upstream in LuxLib, after which this
function should be deleted and the rolled path pointed back at `_sdpa_default` directly. Drift is
caught rather than silent: the chunking tests compare this against the unchunked `_sdpa` layer, so
if LuxLib's conventions move, they go red.
"""
function _sdpa_stable(q::AnyTracedRArray{TQ,N}, k, v, mask, bias) where {TQ,N}
    mat = Reactant.TracedUtils.materialize_traced_array
    q, k, v = mat(q), mat(k), mat(v)
    head_dim, token_dim, num_heads_dim = 1, 3, 2
    @assert size(k, num_heads_dim) == size(q, num_heads_dim) "grouped-query attention is not \
        supported by the rolled chunk path (would need LuxLib's `repeat_inner_gqa`)."

    T = Reactant.unwrapped_eltype(TQ)
    scale = inv(sqrt(T(size(q, head_dim))))
    batch_dims = collect(Int, (N:-1:4..., num_heads_dim))

    # alpha: [B…, H, Nq, Nk] — reversed relative to the generic path's Julia ordering.
    alpha = Reactant.@opcall dot_general(
        Reactant.@opcall(multiply(q, Reactant.@opcall(fill(scale, size(q))))),
        k;
        contracting_dimensions=(Int[head_dim], Int[head_dim]),
        batching_dimensions=(batch_dims, copy(batch_dims)),
    )

    # A rank-r operand in Julia order maps onto alpha's trailing dims reversed: dim i → N+1-i.
    # (LuxLib 1.15.9 hardcodes `[N, N-1]`, which is this rule for a matrix; later versions
    # generalise it exactly this way.)
    _rev_dims(x) = collect(Int, N:-1:(N - ndims(x) + 1))
    if bias !== nothing
        b = mat(bias)
        alpha = Reactant.@opcall add(
            alpha, Reactant.@opcall(broadcast_in_dim(b, _rev_dims(b), collect(Int, size(alpha)))))
    end
    if mask !== nothing
        m = mat(mask)
        alpha = Reactant.@opcall select(
            Reactant.@opcall(broadcast_in_dim(m, _rev_dims(m), collect(Int, size(alpha)))),
            alpha,
            Reactant.@opcall(fill(typemin(eltype(alpha)), size(alpha))))
    end

    # Softmax over the key axis (last, in this ordering). THE ONE CHANGE: the max is a constant
    # shift whose derivative cancels identically, and differentiating it inside a `stablehlo.while`
    # fails to compile ("operand #0 does not dominate this use").
    max_ = _stopgrad(
        Reactant.@opcall(reduce(alpha, typemin(eltype(alpha)), Int64[N], max)))
    max_ = Reactant.@opcall broadcast_in_dim(
        max_, collect(Int, 1:(N - 1)), collect(Int, size(alpha)))
    diff = Reactant.@opcall exponential(Reactant.@opcall(subtract(alpha, max_)))
    sum_ = Reactant.@opcall reduce(diff, eltype(diff)(0), Int64[N], +)
    sum_ = Reactant.@opcall broadcast_in_dim(
        sum_, collect(Int, 1:(N - 1)), collect(Int, size(alpha)))
    scores = Reactant.@opcall divide(diff, sum_)

    x = Reactant.@opcall dot_general(
        scores, v;
        contracting_dimensions=(Int[N], Int[token_dim]),
        batching_dimensions=(collect(Int, 1:(N - 2)), copy(batch_dims)),
    )
    return Reactant.@opcall transpose(x, [N, N - 2, N - 1, (N - 3):-1:1...])
end

"""
    chunked_compile_options(; kwargs...) -> Reactant.CompileOptions

`CompileOptions` under which the rolled chunk loop actually survives to the executable. Pass to
`@compile` / `@code_hlo` as `compile_options=…`; `kwargs` are forwarded to `CompileOptions`.

Emitting a `stablehlo.while` is not enough — two Enzyme-HLO passes each turn it back into the
unchunked graph, and which one fires depends on the chunk count (verified by op-counting the
optimized module at N=64):

  - `greedy_while_loop_batch_fission` sees that the iterations are independent and fuses them into
    a single dot_general batched over all chunks. Same live scores as no chunking. This one fires
    at ANY chunk count and is what gets held off here.
  - the while-loop unroller fires when `nchunks <= Reactant.Compiler.WHILE_UNROLL_THRESHOLD[]`
    (default **4**), giving `nchunks` copies of the body. A global `Ref`, so it cannot be set
    per-compile: set it to `0` yourself. `check_rolled` will catch it.

Both are doing their job — independent iterations normally SHOULD be batched. Rolling is wanted
here only because the whole point is to serialise them.

"Keep `nchunks > 4`" does NOT hold for the REVERSE pass — see [`_warn_unroll_threshold`](@ref).
`checkpointing=true` nests the loop in two, with trip counts of order `sqrt(nchunks)`, so the
reverse's sub-loops sit under the default threshold and get unrolled at essentially any chunk
count. The forward's single loop is unaffected and does follow the `nchunks > 4` rule. Setting
`WHILE_UNROLL_THRESHOLD[] = 0` is the only reliable fix, for either pass.

Reactant ≥ 0.2.278 can exclude the one pass by name; older versions only have the blunter
`disable_loop_raising_passes`, which turns off the surrounding loop-raising family too. Verified
to keep the `while` on 0.2.264 and 0.2.278.
"""
function LuxTriangleAttention.chunked_compile_options(; kwargs...)
    if any(m -> :excluded_passes in Base.kwarg_decl(m), methods(Reactant.CompileOptions))
        return Reactant.CompileOptions(;
            excluded_passes=["greedy_while_loop_batch_fission"], kwargs...)
    end
    return Reactant.CompileOptions(; disable_loop_raising_passes=true, kwargs...)
end

"""
    _warn_unroll_threshold()

Warn once if `Reactant.Compiler.WHILE_UNROLL_THRESHOLD[]` is not `0` when the rolled chunk loop is
about to be traced.

The threshold is documented (see [`chunked_compile_options`](@ref)) as protecting the FORWARD
`stablehlo.while` as long as `nchunks > WHILE_UNROLL_THRESHOLD[]`. It does not protect the
REVERSE: `checkpointing=true` (below) rewrites the reverse into a nested pair of loops whose trip
counts are `O(sqrt(nchunks))`, not `nchunks` — so both nested bounds land under the default
threshold of `4` for any chunk count worth using, and the reverse is unrolled regardless of
`nchunks`. Verified: `nchunks` of 5, 6, 8 and 10 all fully unroll the reverse at the default
threshold (`4`), while all four stay rolled once `WHILE_UNROLL_THRESHOLD[]` is set to `0`.

An unrolled reverse silently gives back the entire memory bound chunking exists for — worse, on
some Reactant versions (verified: 0.2.264) it does not even compile, hanging indefinitely instead.
`WHILE_UNROLL_THRESHOLD[] = 0` is a global `Ref` and cannot be set per-compile via
`CompileOptions`; it must be set by the caller before `@compile`/`@code_hlo`.
"""
function _warn_unroll_threshold()
    thr = Reactant.Compiler.WHILE_UNROLL_THRESHOLD[]
    thr == 0 || @warn """
    `Reactant.Compiler.WHILE_UNROLL_THRESHOLD[]` is $thr (not `0`) while compiling chunked \
    attention. This does not protect the REVERSE pass: checkpointing nests the loop in two, with \
    trip counts of order `sqrt(nchunks)`, so the reverse is unrolled at essentially any chunk \
    count — silently negating the memory bound chunking exists for (and it may also unroll the \
    forward, if `nchunks <= WHILE_UNROLL_THRESHOLD[]`). Set \
    `Reactant.Compiler.WHILE_UNROLL_THRESHOLD[] = 0` before compiling the backward pass.
    """ maxlog = 1
end

"""
    check_rolled(mod) -> Bool

Whether an optimized module from `@code_hlo` still contains a `stablehlo.while`. Cheap guard
against the chunking having been silently optimized away; see [`chunked_compile_options`](@ref).
"""
LuxTriangleAttention.check_rolled(mod) = occursin("stablehlo.while", string(mod))

# ── Chunked attention, rolled ────────────────────────────────────────────────────────────────────
#
# The base package chunks with an ordinary Julia loop, which bounds peak memory when executed
# sequentially. Inside a Reactant trace that loop is UNROLLED, and the chunks are mutually
# independent, so XLA schedules them concurrently and keeps every slice's scores live — measured to
# give no memory benefit at all, forward or reverse (N=768 forward and N=400 VJP both still
# exhausted 32 GB with chunking on).
#
# `@trace` lowers the loop to a single `stablehlo.while`, which forces the chunks to run one at a
# time: only one slice's scores are live. `checkpointing=true` additionally lets Enzyme recompute
# slices in the backward instead of storing them all, which is what bounds the REVERSE peak.
#
# Emitting the `while` is only half of it: the optimizer removes it again unless compiled under
# `chunked_compile_options()` (above). Anything calling this for the memory bound must pass them.
#
# Constraints inherited from `stablehlo.while` (same ones AF2's fold loop documents): a static
# bound, one integer induction variable, uniform per-iteration shapes, and writes via `=` (a traced
# index lowers to dynamic-update-slice; broadcasting into a view indexed by a traced scalar does
# not trace). Uniform shapes mean the chunk must divide the axis exactly — otherwise fall back to
# the unrolled form, which handles any chunk size.
function LuxTriangleAttention._chunked_sdpa(
    chunk_size::Int, q::AnyTracedRArray, k, v, mask, bias
)
    L = size(q, 4)
    chunk_size >= L && return LuxTriangleAttention._sdpa(q, k, v, mask, bias)
    L % chunk_size == 0 ||
        return LuxTriangleAttention._chunked_sdpa_unrolled(chunk_size, q, k, v, mask, bias)

    _warn_unroll_threshold()
    nchunks = L ÷ chunk_size

    # Split the chunked axis in two: `[…, L, …]` → `[…, chunk_size, nchunks, …]`. Column-major makes
    # this exact — element `l = (c-1)*chunk_size + j` sits at `(j, c)` — and it turns the per-chunk
    # slice into a SCALAR index on the new axis, which is the shape `stablehlo.while` can carry.
    qc, _ = _split_axis(q, chunk_size, nchunks, L)
    kc, _ = _split_axis(k, chunk_size, nchunks, L)
    vc, _ = _split_axis(v, chunk_size, nchunks, L)
    # `Val` and not a plain `Bool`: any ordinary local used inside the loop body is promoted to a
    # `TracedRNumber` by `@trace`, and a traced flag cannot select a method. As a singleton type it
    # stays a trace-time constant.
    mc, m_split = _split_axis(mask, chunk_size, nchunks, L)
    bc, b_split = _split_axis(bias, chunk_size, nchunks, L)
    m_tag, b_tag = Val(m_split), Val(b_split)

    # Preallocated so the buffer is part of the graph before the loop writes into it.
    out = similar(qc)
    fill!(out, zero(eltype(qc)))

    @trace checkpointing = true for c in 1:nchunks
        out[_chunk_index(out, c)...] = _sdpa_stable(
            qc[_chunk_index(qc, c)...], kc[_chunk_index(kc, c)...], vc[_chunk_index(vc, c)...],
            _take_chunk(mc, c, m_tag), _take_chunk(bc, c, b_tag))
    end

    return reshape(out, size(q))
end

# `_chunked_sdpa(::Nothing, ...)` needs no traced override: the base package's generic method
# (`_chunked_sdpa(::Nothing, q, k, v, mask, bias) = _sdpa(q, k, v, mask, bias)` in
# `src/layers/chunk.jl`) already reaches `_sdpa_default` for a traced array, since `_sdpa` itself
# has no traced-specific override either.

# `[…, L, …]` → `[…, chunk_size, nchunks, …]`, but only for operands that actually span the chunked
# axis. A size-1 axis is a broadcast dim (what `prep_bias` produces for `:qk`) and is passed whole.
# Returns `(operand, was_split)` — the flag cannot be recovered from `ndims` afterwards, since an
# unsplit 5-D bias and a split 5-D mask look identical.
_split_axis(::Nothing, _, _, _) = (nothing, false)
function _split_axis(a::AbstractArray, chunk_size::Int, nchunks::Int, L::Int)
    # Unsplit operands are captured into the loop closure too, so they must be real buffers as well.
    size(a, 4) == L || return (_materialize(a), false)
    sz = size(a)
    return (_materialize(reshape(a, sz[1], sz[2], sz[3], chunk_size, nchunks, sz[5:end]...)), true)
end

# `reshape` over a traced array (q/k/v arrive from `_prep_qkv` as views) stays LAZY, and a
# `stablehlo.while` cannot carry a `Base.ReshapedArray`: its `dims` field would have to become
# traced `Int`s, which fails as `NoFieldMatchError`. Force a real buffer — same fix as AF2's fold
# loop (`_materialize_carry`).
_materialize(x::Base.ReshapedArray) = copy(x)
_materialize(x) = x

# Index tuple selecting chunk `c`: `[:, :, :, :, c, ..]`.
_chunk_index(a::AbstractArray, c) =
    (ntuple(_ -> Colon(), 4)..., c, ntuple(_ -> Colon(), ndims(a) - 5)...)

# Operands left unsplit (broadcast axes) are used whole; split ones are indexed at `c`.
_take_chunk(::Nothing, _, _) = nothing
_take_chunk(a::AbstractArray, c, ::Val{true}) = a[_chunk_index(a, c)...]
_take_chunk(a::AbstractArray, _, ::Val{false}) = a
