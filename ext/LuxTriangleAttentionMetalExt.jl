module LuxTriangleAttentionMetalExt

import Metal: AS, GPUCompiler

using Metal
using LuxTriangleAttention

const LTA = LuxTriangleAttention

dptr(::Type{T}) where {T} = Core.LLVMPtr{T,AS.Device}
const PL = Core.LLVMPtr{Float32,AS.Device}   # `lse` is never narrowed
const PB = Core.LLVMPtr{Bool,AS.Device}

"""
    supported_eltypes() -> Tuple{Vararg{DataType}}

Storage element types this backend will actually run.

The kernel is written generically over the storage type (accumulation is always `Float32` — see
note 3 in `src/kernels/flash/forward.jl`), and `Float16` would compile here today. It is
deliberately NOT listed: a path that has only ever been executed at `Float32` is untested, not
working, and the parity tolerances at half precision have not been established. Widening this tuple
is a decision to be made per dtype AFTER running the reference comparison, not a default.

`BFloat16` additionally needs a capability probe rather than a flat allowance — Apple requires newer
Metal for it than for `Float16`, so device support cannot be assumed from the fact that Metal.jl can
emit `bf16`.
"""
supported_eltypes() = (Float32,)

function _check_eltype(::Type{T}) where {T}
    T in supported_eltypes() && return nothing
    throw(ArgumentError(
        "storage eltype $T is not enabled for the Metal flash kernel (enabled: " *
        "$(join(supported_eltypes(), ", "))). The kernel is generic over storage precision and " *
        "$T may well compile, but it has not been validated against the reference — see " *
        "`supported_eltypes`."))
end

# The kernel now lives in `LuxTriangleAttention` itself, so it is reached by ordinary name
# resolution — the module-registry indirection this file used to need is gone.
cache_dir() = LTA.binary_cache_dir("metal")

"""
    build(T, dim, tg, kt; static_n=nothing, heads=1, force=false) -> (path, entry)

Compile one configuration of the shared kernel to a `.metallib`, or return the cached one.

`T`, `dim`, `tg` and `kt` are `Val` type parameters, so each is a separate compilation — which is the
point: the tile loop unrolls against them and the query/accumulator tuples become fixed-size
register residents. **`n` and the head count are runtime arguments and deliberately absent from the
cache key**, so one binary covers every sequence length.

`static_n` bakes a sequence length in, which makes KA's context a ghost type and removes it from the
ABI. Measured on Metal it is within 1% of the `n`-independent form at every size — i.e. inside
noise — so it exists for completeness and is not the default. Cold compilation is 0.17-0.85 s
either way.

The entry name also carries `LuxTriangleAttention.kernel_source_tag()`, so editing a kernel produces
a different filename rather than a stale cache hit — see that function for why the configuration
alone is not a sufficient key. Binaries built from superseded sources are removed as they are
replaced, by [`_prune_stale`](@ref).
"""
function build(::Type{T}, dim::Int, tg::Int, kt::Int;
    static_n::Union{Nothing,Int}=nothing, heads::Int=1, force::Bool=false) where {T}
    dir = cache_dir()
    mkpath(dir)
    # The storage type MUST be in the cache key. Without it a Float16 build would silently be served
    # for a Float32 request — same filename, different ABI, wrong answers with no error.
    tag = lowercase(string(nameof(T)))
    stem = static_n === nothing ? "lta_ka_$(tag)_d$(dim)_t$(tg)_k$(kt)" :
           "lta_kaS_$(tag)_d$(dim)_t$(tg)_k$(kt)_n$(static_n)_h$(heads)"
    entry = "$(stem)_$(LTA.kernel_source_tag())"
    path = joinpath(dir, entry * ".metallib")
    (!force && isfile(path)) && return (path, entry)
    _prune_stale(dir, stem, entry)

    ctx, _, _ = LTA.kernel_context(MetalBackend(),
        something(static_n, 512), heads, tg; static=(static_n !== nothing))
    tt = LTA.kernel_signature(typeof(ctx), dptr(T), PL, PB, T, dim, tg, kt)
    obj = LTA.kernel_object(MetalBackend(), tg;
        ndrange=(static_n === nothing ? nothing :
                 LTA.launch_geometry(static_n, heads, tg).ndrange))
    job = GPUCompiler.CompilerJob(GPUCompiler.methodinstance(typeof(obj.f), tt),
        Metal.compiler_config(Metal.device(); name=entry, kernel=true))
    write(path, Metal.compile(job).metallib)
    return (path, entry)
end

"""
    _prune_stale(dir, stem, keep)

Delete binaries for configuration `stem` that were built from superseded kernel sources.

Called only when a binary for `stem` is about to be (re)built, so it never runs on the cache-hit
path. Without it every kernel edit would leave its predecessor behind forever — harmless to
correctness once the source tag is in the name, but the cache is a few hundred KB per configuration
and there are sixteen of them per dtype.
"""
function _prune_stale(dir, stem::AbstractString, keep::AbstractString)
    for f in readdir(dir)
        # `stem * ".metallib"` catches binaries from before the source tag existed; the underscore
        # form catches every tagged one. Matching a bare `stem` prefix would not do: `..._k16` is a
        # prefix of `..._k160` should that tile size ever be added.
        (f == stem * ".metallib" || startswith(f, stem * "_")) || continue
        f == keep * ".metallib" && continue
        rm(joinpath(dir, f); force=true)
    end
    return nothing
end

"""
    build_bwd(T, dim, tg, kt; force=false) -> (; dq, dk, dv)

Compile the three backward kernels — [`LuxTriangleAttention.flash_bwd_dq!`](@ref),
[`LuxTriangleAttention.flash_bwd_dk!`](@ref) and [`LuxTriangleAttention.flash_bwd_dv!`](@ref) — for
one configuration, same caching as [`build`](@ref). Kept as one function (rather than three) because
every caller wants the whole backward together.

Returns a `NamedTuple` of `(path, entry)` pairs rather than a flat tuple: at two kernels a
positional 4-tuple was still readable, at three it is six values whose order nothing checks.
"""
function build_bwd(::Type{T}, dim::Int, tg::Int, kt::Int; force::Bool=false) where {T}
    dir = cache_dir()
    mkpath(dir)
    tag = lowercase(string(nameof(T)))
    src = LTA.kernel_source_tag()

    # `LTA.kernel_object` is hardcoded to `flash_fwd!` (it is the forward's own helper); the launch
    # geometry it builds is identical for all of these kernels (same `(tg,1,1)` group shape), but the
    # KA kernel OBJECT itself must name the right function, so each is constructed directly here.
    specs = (dq=(sig=LTA.kernel_signature_dq, kern=LTA.flash_bwd_dq!),
        dk=(sig=LTA.kernel_signature_dk, kern=LTA.flash_bwd_dk!),
        dv=(sig=LTA.kernel_signature_dv, kern=LTA.flash_bwd_dv!))

    return NamedTuple(k => begin
        stem = "lta_ka_bwd_$(k)_$(tag)_d$(dim)_t$(tg)_k$(kt)"
        entry = "$(stem)_$(src)"
        path = joinpath(dir, entry * ".metallib")
        if force || !isfile(path)
            _prune_stale(dir, stem, entry)
            ctx, _, _ = LTA.kernel_context(MetalBackend(), 512, 1, tg)
            tt = spec.sig(typeof(ctx), dptr(T), PL, PB, T, dim, tg, kt)
            obj = spec.kern(MetalBackend(), (tg, 1, 1))
            job = GPUCompiler.CompilerJob(GPUCompiler.methodinstance(typeof(obj.f), tt),
                Metal.compiler_config(Metal.device(); name=entry, kernel=true))
            write(path, Metal.compile(job).metallib)
        end
        (path, entry)
    end for (k, spec) in pairs(specs))
end

# The one thing `flash_bwd_dq!` needs that KernelAbstractions cannot express: an atomic add on the
# raw pointer `dbias` (see `_atomic_fadd!`'s docstring in `flash/backward.jl`). Metal has a native
# `atomic_fetch_add_explicit` for `Float32` device pointers (Metal 3.0+); the byte-offset arithmetic
# here is exactly what `pointer(A, idx)` does for any device array, so this is the same operation
# `@atomic A[idx] += val` performs on a real `MtlArray` — just spelled out for a raw pointer instead.
LTA._atomic_fadd!(ptr::Core.LLVMPtr{Float32,AS.Device}, idx::Integer, val::Float32) =
    Metal.atomic_fetch_add_explicit(ptr + (Int32(idx) - Int32(1)) * Int32(sizeof(Float32)), val)

include("metal/flash.jl")
include("metal/tune.jl")
# The entry-driven launcher: runs a registry entry's stages without knowing which kernel it is.
include("metal/launch.jl")

const _AVAILABLE = Ref{Union{Nothing,Bool}}(nothing)
function _available()
    v = _AVAILABLE[]
    v === nothing || return v
    ok = try
        Metal.functional() && Metal.device() !== nothing
    catch
        false
    end
    _AVAILABLE[] = ok
    return ok
end

# `backend_of` is how `_sdpa_kernel` tells host from device without dispatching on the array type
# directly — see `src/kernels/backend.jl`.
LTA.backend_of(::MtlArray) = LTA.MetalBackendTag()

"""
    __init__()

Install this backend's hooks and announce that Metal is available.

Nothing is REGISTERED here any more. The built-in kernel is compiled and registered by
`tune_flash_kernel!`, on demand or ahead of time, so that a `.metallib` built in an earlier session
is reused rather than the registry being rebuilt from closures on every start. What this installs
are the seams `tune_flash_kernel!` and the Reactant extension reach the backend through — neither
can import this module directly, and neither is guaranteed to be loaded.

Adding `:metal` to `_LOADED_BACKENDS` is what makes `use_kernel!` able to reject a CUDA kernel here
at the point of selection instead of at the first launch.
"""
function __init__()
    LTA._DEVICE_AVAILABLE[] = _available
    LTA._DEVICE_BINARY[] = _binary
    LTA._DEVICE_BINARY_BWD[] = _binary_bwd
    LTA._KA_BACKEND_OBJECT[] = () -> MetalBackend()
    _available() && push!(LTA._LOADED_BACKENDS, :metal)
    return nothing
end

end # module
