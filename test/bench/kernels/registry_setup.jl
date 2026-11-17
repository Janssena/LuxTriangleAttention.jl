# Kernel setup shared by the kernel benchmarks and `trifast_correctness.jl`: both kernels under test
# — our KernelAbstractions flash kernel (`:default`) and trifast's Triton-generated metallibs
# (`:trifast`) — are kernels of the `SDPA` op, picked with `kernel_named` and run through the SAME
# `KernelDispatch.forward` / `sdpa_bwd` path: same operands, same layout. (Not through `use_kernel!`: a switch
# only takes effect from the next top-level expression, so it cannot be made and used inside one function.)
#
# INCLUDE THIS BEFORE `using LuxTriangleAttention`. It points the kernel cache at a private
# directory (`KERNELDISPATCH_CACHE`). Tuned variants persist, and
# this setup deliberately holds exactly one tuning of `:default` (see `tune_default_fixed!`); doing
# that in your real cache would overwrite the per-length tuning your models use.
#
# Matched configuration. trifast is compiled once, at fixed 32×32 tiles and 1024 threads, and that
# one binary serves every `n`. Our kernel is held to the same rule: tuned ONCE at a fixed reference
# length and reused for every `n`, rather than re-tuned per `n`.
#
# trifast doubles as the worked example of bringing your own attention kernel (see
# `docs/custom-kernels.md`): a kernel type, its tiles, and `forward`/`backward` methods that hand each
# binary its arguments in the order it takes them. The metallibs are built by
# `trifast.metal.build(dim=32)` in ../trifast from `triton-metal.py`, whose kernels read this
# package's `[hd, H, Nq, Nj, B]` arrays in place.

ENV["KERNELDISPATCH_CACHE"] = get(ENV, "LTA_BENCH_KERNEL_CACHE",
    joinpath(@__DIR__, ".bench_kernel_cache"))

using LuxTriangleAttention
import KernelDispatch
using KernelDispatch: KernelBinary, OutArray, call_binary

"The single length our kernel is tuned at, for every `n` (see this file's header)."
const KA_TUNE_N = parse(Int, get(ENV, "LTA_BENCH_TUNE_N", "512"))

"""
    tune_default_fixed!(; n=KA_TUNE_N, dim=32) -> variants

Make `:default` hold exactly ONE tuning, at length `n`, so every call — whatever its own length —
resolves to it (`nearest` falls back across the size keys, never across dtype, dim or bias/mask
presence). Anything else is dropped first: a variant tuned at another length would be the closer
match for some `n` and quietly reintroduce per-length tuning. One tuning is four variants, one per
bias/mask presence.
"""
function tune_default_fixed!(; n::Integer=KA_TUNE_N, dim::Integer=32)
    LTA = LuxTriangleAttention
    op, k = LTA.SDPA(), LTA.FlashScalar()
    empty!(KernelDispatch.variants(op, k))
    rm(KernelDispatch.kernel_dir(op, k); recursive=true, force=true)
    vs = LTA.tune_flash_kernel!(n; dim)
    length(vs) == 4 || error("`:default` holds $(length(vs)) variants after a fixed tune; expected 4")
    return vs
end

"Directory `trifast.metal.build(dim=32)` writes to (`TRIFAST_METAL_CACHE` overrides the cache root)."
const TRIFAST_DIR = joinpath(get(ENV, "TRIFAST_METAL_CACHE",
        joinpath(homedir(), "Library", "Caches", "trifast")),
    "trifast_d32_j32_k32")

include(joinpath(@__DIR__, "..", "..", "support", "trifast_kernel.jl"))   # the `Trifast` kernel type and its methods

"The kernel object registered as `name`: `:default` is the package's own flash kernel, `:trifast` the one above."
function kernel_named(name::Symbol)
    name === :default && return LuxTriangleAttention.FlashScalar()
    name === :trifast && return Trifast()
    throw(ArgumentError("no kernel named :$name; expected :default or :trifast"))
end

"""
    register_trifast!(; dir=TRIFAST_DIR, dim=32)

Register trifast's four metallibs as the `:trifast` kernel's variant for `(Float32, dim)`.
`threadgroup = 1024` is what triton-msl's generic lowering reports for every one of these kernels
(`metadata.block_size`, in trifast's `trifast_d32_j32_k32.json`): a property of the build, to
re-check after rebuilding with a different triton-msl. Each group covers 32 query rows (`tile =
(32, 1, 1)`); `_bwd_b` covers a 32×32 block of the pair bias (`tile = (32, 32, 1)`).
"""
function register_trifast!(; dir::AbstractString=TRIFAST_DIR, dim::Integer=32)
    f(s) = joinpath(dir, "$s.metallib")
    for s in ("_fwd", "_bwd_q", "_bwd_kv", "_bwd_b")
        isfile(f(s)) || error("trifast metallib not found at $(f(s)); run " *
                              "`trifast.metal.build(dim=$dim)` in ../trifast first")
    end
    bin(s, stage; tile=(TRIFAST_TILE, 1, 1)) =
        KernelBinary(f(s); threadgroup=1024, tile, signature=TRIFAST_SIGNATURES[stage])
    op = LuxTriangleAttention.SDPA()
    KernelDispatch.register_kernel!(op, Trifast())
    KernelDispatch.add_variant!(op, Trifast(), (:Float32, Int(dim));
        fwd=bin("_fwd", :fwd), dq=bin("_bwd_q", :dq), dkv=bin("_bwd_kv", :dkv),
        dbias=bin("_bwd_b", :dbias; tile=(TRIFAST_TILE, TRIFAST_TILE, 1)))
    return Trifast()
end
