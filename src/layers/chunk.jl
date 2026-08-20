"""
    _chunked_sdpa(chunk_size, q, k, v, mask, bias)

Scaled dot-product attention, optionally split into `chunk_size` slices along dim 4 of `q` — the
first axis the attention batches over, since dim 3 is the token axis being attended.

For a pair input `[C, Ni, Nj, B]` dim 4 is `Nj` (a TOKEN axis, not the sample batch), and the
scores are `[Nq, Nk, H, Nj, B]` = O(N³·H) — ~1 GiB per op at N=400. Splitting it bounds the live
score tensor to `chunk_size · N² · H` with no change to the result, and does so at `B == 1`.
Mirrors `chunk_size_tri_attn` in the openfold/boltz references.

Operands are sliced only when they actually span the chunked axis (`size(·, 4) == L`); a broadcast
axis of size 1 — which is what `prep_bias` produces for the `:qk` layout — is passed through whole.
"""
_chunked_sdpa(::Nothing, q, k, v, mask, bias) = _sdpa(q, k, v, mask, bias)

function _chunked_sdpa(chunk_size::Int, q, k, v, mask, bias)
    _warn_chunking_on_gpu(backend_of(q))
    return _chunked_sdpa_unrolled(chunk_size, q, k, v, mask, bias)
end

"""
    _warn_chunking_on_gpu(backend)

Say once, on a device backend, that chunking is the wrong tool here.

`chunk_size` exists because the host path materialises the whole `[Nq, Nk, H, Nj, B]` score tensor
and has no other way to bound it. A fused kernel tiles the keys internally and never materialises it
at all, so on a GPU chunking pays the memory cost it was meant to avoid AND serialises work the
device would otherwise run at once. Reaching here with device arrays almost always means
`use_kernel` was left at its default rather than that chunking was wanted.

Uses `backend_of` rather than a `GPUArrays` dependency: the trait already distinguishes `MtlArray`,
`CuArray` and `ROCArray`, and it additionally covers a traced array on a GPU client, which an array
type check would miss entirely.
"""
_warn_chunking_on_gpu(::KernelBackend) = nothing
_warn_chunking_on_gpu(::GPUBackend) = @warn """
`chunk_size` is set and these arrays are on a GPU, where chunking is not the fastest path: it
bounds the score tensor by slicing, which a fused kernel avoids entirely by tiling the keys.
Consider `Attention(...; use_kernel=static(true))` instead, and see `tune_flash_kernel!`.
""" maxlog = 1

"""
    _chunked_sdpa_unrolled(chunk_size, q, k, v, mask, bias)

Chunked attention as an ordinary Julia loop. Executed sequentially it bounds the live score tensor,
which is the intended win on the native path — measured at N=512 on one `PairformerLayer`: 3.2 GB
less peak for ~12% more time.

Under `Reactant.@compile` this loop is UNROLLED into the graph, and because the chunks are mutually
independent XLA is free to schedule them concurrently and keep every slice live, completely negating 
any memory benefits. The Reactant extension therefore overrides `_chunked_sdpa` with a rolled `@trace` 
loop; this method stays the native path and the fallback for chunk sizes the rolled form cannot take.
"""
function _chunked_sdpa_unrolled(chunk_size::Int, q, k, v, mask, bias)
    L = size(q, 4)
    chunk_size >= L && return _sdpa(q, k, v, mask, bias)   # nothing to gain

    parts = map(Iterators.partition(1:L, chunk_size)) do r
        _sdpa(selectdim(q, 4, r), selectdim(k, 4, r), selectdim(v, 4, r),
            _chunk_operand(mask, r, L), _chunk_operand(bias, r, L))
    end
    return reduce((a, b) -> cat(a, b; dims=4), parts)
end

_chunk_operand(::Nothing, _, _) = nothing
_chunk_operand(a::AbstractArray, r, L) = size(a, 4) == L ? selectdim(a, 4, r) : a
