# Which fused kernel runs, and how a user substitutes their own.
#
# There is exactly ONE kernel in use at a time, named by `LTA_KERNEL[]`. This file owns the table of
# what is available (the registry), the selection (`use_kernel!`), and the calling convention that
# lets a launcher bind buffers to a kernel it has never seen.
# ## The ABI contract
#
# A launcher must bind buffers to a binary whose source it has never seen. The contract is that the
# buffer table is DERIVED, never specified:
#
#     [ preamble... ][ outputs ][ inputs ][ scalars ]
#
# with each group in a fixed canonical order (`_ROLE_RANK`). A stage declares WHICH roles it
# touches; it never declares their ORDER. This is the whole reason the contract is safe to write by
# hand: a mis-ordered slot table binds without complaint and produces wrong numbers rather than an
# error, so the ordering is taken out of the user's hands entirely.
#
# A third-party kernel with a different argument order (e.g. trifast's Triton kernels take a pointer
# followed by one explicit stride per dimension) is adapted with a thin shim kernel that reorders 
# arguments; not by teaching the registry a bespoke slot table.

"""
    RESERVED_KERNEL_NAME

The one entry name [`register_kernel!`](@ref) refuses. `:default` is our own KernelAbstractions
flash kernel, written only by [`tune_flash_kernel!`](@ref), which compiles it for a specific
sequence length. Letting a user overwrite it would leave no way back to a working kernel short of
clearing the cache by hand.
"""
const RESERVED_KERNEL_NAME = :default

"""
    _ROLE_RANK

Canonical order of every role a stage can touch, as `role => rank`. Sorting a stage's declared
roles by this is what turns a set of roles into a buffer table — see this file's header for why the
order is ours and not the caller's.

Three groups, kept in one table so a role can never be classified two ways:

  * **outputs** (`o` … `dbias`) — buffers a stage WRITES. The runtime allocates them.
  * **inputs** (`q` … `delta_in`) — buffers a stage READS. Either a primal argument or an output
    written by an EARLIER stage (`delta` is the only one of those in practice). The incoming
    cotangent `dO` sits BEFORE the residuals, matching `flash_bwd_dq!`/`dk!`/`dv!`'s own argument
    order — this ordering is the contract third-party kernels are shimmed to, so it is fixed by what
    the shipped kernels already do rather than chosen freely.
  * **scalars** (`scale` … `dim`) — passed by value, after every buffer.

`nh` and `mheads` are deliberately BOTH here rather than collapsed into `h`. They coincide for
LuxTriangleAttention-layout calls (both equal the head count) but not for packed `[dim, n, n, bh]`
callers, where `nh = 1` and `mheads` is the true head count — see `flash_fwd!`'s addressing comment.
Collapsing them would silently mis-address the mask on the packed path.
"""
const _ROLE_RANK = Dict{Symbol,Int}(
    # outputs
    :o => 100, :lse => 101, :delta => 102,
    :dq => 110, :dk => 111, :dv => 112, :dbias => 113,
    # inputs
    :q => 200, :k => 201, :v => 202, :bias => 203, :mask => 204,
    :dO => 205, :o_in => 206, :lse_in => 207, :delta_in => 208,
    # scalars
    :scale => 300, :neg_inf => 301, :n => 302, :nh => 303, :mheads => 304,
    :h => 305, :dim => 306,
)

const _OUTPUT_ROLES = Set([:o, :lse, :delta, :dq, :dk, :dv, :dbias])
const _INPUT_ROLES = Set([:q, :k, :v, :bias, :mask, :o_in, :lse_in, :delta_in, :dO])
const _SCALAR_ROLES = Set([:scale, :neg_inf, :n, :nh, :mheads, :h, :dim])

"""
    _GRID_AXES

Grid extents a stage may ask for, resolved against the problem size at launch.

`:ka_ngroups` is the whole 1-D group count KernelAbstractions' own launch geometry produces, and is
what our kernel uses — see `FLASH_GRID` for why a 3-D grid is not merely unidiomatic here but
inexpressible. `:cdiv_n_bj`/`:cdiv_n_bk` are the tile counts a Triton-style kernel parallelises
over, `:n` and `:bh` the plain extents, `:one` a filler for unused axes.
"""
const _GRID_AXES = (:cdiv_n_bj, :cdiv_n_bk, :n, :bh, :one, :ka_ngroups)

# `o`/`lse`/`delta` are both produced (by some stage) and consumed (by a later one). They need
# distinct names so a role can be classified without knowing which stage is asking, hence the
# `_in` suffix on the reading side, and this map back to the producing name.
const _INPUT_SOURCE = Dict(:o_in => :o, :lse_in => :lse, :delta_in => :delta)

_role_rank(r::Symbol) = get(_ROLE_RANK, r) do
    throw(ArgumentError("unknown kernel role `:$r`. Valid roles: " *
        join(sort(collect(keys(_ROLE_RANK))), ", "))
    )
end

"""
    KernelStage

One compiled kernel launch within an entry.

A backward is not one launch, and different implementations split it differently: ours is
`dq`(+`dbias`) / `dk` / `dv`, while trifast's is `dq`(+`delta`) / `dk`+`dv` / `dbias`. Rather than
forcing either split on the other, an entry carries an ORDERED list of stages and each declares
what it writes.

# Fields
- `name`: identifier within the entry (`:fwd`, `:dq`, `:dkv`, …). Only `:fwd` is meaningful to the
  forward path; the rest are run in order by the backward.
- `file`: basename inside the entry's cache directory.
- `entry`: the symbol to look up inside the binary.
- `outputs`, `inputs`, `scalars`: roles this stage touches, as SETS — order is imposed by
  [`abi_order`](@ref), never taken from these vectors.
- `grid`: three symbols from `(:cdiv_n_bj, :cdiv_n_bk, :n, :bh, :one)`, resolved against the
  problem size at launch.
- `threadgroup`: threads per group.
"""
struct KernelStage
    name::Symbol
    file::String
    entry::String
    outputs::Vector{Symbol}
    inputs::Vector{Symbol}
    scalars::Vector{Symbol}
    grid::NTuple{3,Symbol}
    threadgroup::Int
end

"""
    KernelEntry

One registered kernel implementation: every stage, and everything needed to launch them.

# Fields
- `name`: unique in the registry. Re-registering a name replaces it.
- `backend`/`format`: `:metal`/`:cuda`/`:rocm` and `:metallib`/`:cubin`/`:fatbin`/`:hsaco`,
  both inferred from the registered files' extension. [`use_kernel!`](@ref) checks `backend`
  against what is actually loaded.
- `layout`: the axis order the binaries were compiled for, e.g. `(:hd, :H, :Nq, :Nj, :B)` for ours
  or `(:bh, :n, :n, :dim)` for trifast. The launcher permutes into it.
- `rowmajor`: whether `layout` is written fastest-axis-last.
- `keys`: which of `(:dtype, :dim, :nbucket)` the binaries vary over. Ours is `(:dtype, :dim,
  :nbucket)`; a kernel that takes `n` as a runtime argument omits `:nbucket`.
- `variants`: key tuple → ordered stages. One entry can hold many compiled configurations.
- `preamble`: leading `:bytes` slots, declared per entry. Exists only for our KernelAbstractions
  kernel's `(kernel_state, KA_context)` prefix; empty for everyone else.
- `description`: shown by [`list_kernels`](@ref).
"""
struct KernelEntry
    name::Symbol
    backend::Symbol
    format::Symbol
    layout::Vector{Symbol}
    rowmajor::Bool
    keys::Vector{Symbol}
    variants::Dict{Tuple,Vector{KernelStage}}
    preamble::Vector{Symbol}
    description::String
end

"""
    abi_order(entry, stage) -> Vector{Pair{Symbol,Symbol}}

The buffer table for one stage, as `kind => role` in binding order, where `kind` is `:bytes`,
`:out`, `:in` or `:scalar`.

This is THE contract. A launcher binds exactly this, in exactly this order; a compiled kernel must
have been written (or shimmed) to expect it. See this file's header for why the order is derived
from a fixed rank table rather than taken from the caller.
"""
function abi_order(entry::KernelEntry, stage::KernelStage)
    slots = Pair{Symbol,Symbol}[]
    for p in entry.preamble
        push!(slots, :bytes => p)
    end
    for r in sort(stage.outputs; by=_role_rank)
        push!(slots, :out => r)
    end
    for r in sort(stage.inputs; by=_role_rank)
        push!(slots, :in => r)
    end
    for r in sort(stage.scalars; by=_role_rank)
        push!(slots, :scalar => r)
    end
    return slots
end

const _KERNELS = Dict{Symbol,KernelEntry}()

"""
    LTA_KERNEL

The kernel currently selected, as a `StaticSymbol`. Never empty: it starts at
[`RESERVED_KERNEL_NAME`](@ref) (`:default`, our own flash kernel) and [`use_kernel!`](@ref) is the
only thing that moves it.

A `StaticSymbol` rather than a plain `Symbol` so the selection is a TYPE at the point it is
dispatched on. Reading the `Ref` is type-unstable and so costs one dynamic dispatch per
`_sdpa_kernel` call — negligible beside a kernel launch, and it buys specialisation inside the
method that receives it.

Deliberately NOT persisted. The registry survives a restart; the selection does not, so a stale
choice made in one session cannot silently change results in an unrelated one.
"""
const LTA_KERNEL = Ref{StaticSymbol}(static(RESERVED_KERNEL_NAME))

"""
    current_kernel() -> Symbol

The name of the selected kernel. See [`LTA_KERNEL`](@ref).
"""
current_kernel() = dynamic(LTA_KERNEL[])

"""
    kernel_entry([name]) -> KernelEntry

The registered entry for `name`, defaulting to the selected kernel. Throws if absent, naming what
IS registered — an absent entry almost always means the extension that registers it never loaded,
and the list is the fastest way to see that.
"""
kernel_entry() = kernel_entry(current_kernel())

kernel_entry(name::Symbol) = get(_KERNELS, name) do
    throw(
        ArgumentError("no kernel registered under `:$name`. Registered: " *
            (isempty(_KERNELS) ? "(none)" : join(sort(collect(keys(_KERNELS))), ", "))
        )
    )
end

"""
    has_kernel(name) -> Bool

Whether `name` is registered. Non-throwing counterpart to [`kernel_entry`](@ref).
"""
has_kernel(name::Symbol) = haskey(_KERNELS, name)

const _FORMAT_BACKEND = Dict(
    ".metallib" => (:metallib, :metal),
    ".cubin" => (:cubin, :cuda),
    ".fatbin" => (:fatbin, :cuda),
    ".hsaco" => (:hsaco, :rocm),
    ".co" => (:hsaco, :rocm),
)

"""
    _classify(paths) -> (format, backend)

Infer the binary format and backend from the registered files' extension, requiring all of them to
agree. Mixing a `.metallib` with a `.cubin` in one entry is a mistake with no sensible
interpretation, and catching it here is much cheaper than at launch.
"""
function _classify(paths)
    seen = Set{Tuple{Symbol,Symbol}}()
    for p in paths
        ext = lowercase(splitext(p)[2])
        haskey(_FORMAT_BACKEND, ext) || throw(ArgumentError(
            "`$p` has extension `$ext`, which is not a kernel binary format. Supported: " *
            join(sort(collect(keys(_FORMAT_BACKEND))), ", ")))
        push!(seen, _FORMAT_BACKEND[ext])
    end
    length(seen) == 1 || throw(ArgumentError(
        "all files in one kernel entry must share a binary format; got " *
        join(sort([string(f) for (f, _) in seen]), ", ")))
    return first(seen)
end

"""
    nbucket(n) -> Int

The next power of two at or above `n`.

Kernels that bake the sequence length into a compile-time constant do it at this granularity —
trifast's `CLOSEST_N` is exactly `2^ceil(log2(n))` — so bucketing here means one binary serves a
range of `n` rather than one binary per `n`. Our own kernel takes `n` as a runtime argument and
omits `:nbucket` from its keys entirely; it is the TUNED configuration, not the code, that varies
with length.
"""
nbucket(n::Integer) = 1 << max(0, ceil(Int, log2(max(n, 1))))

"""
    variant_key(entry, T, dim, n) -> Tuple

The lookup key for one problem, holding only the fields `entry.keys` declares. An entry that varies
over nothing has key `()` and a single variant.
"""
function variant_key(entry::KernelEntry, ::Type{T}, dim::Integer, n::Integer) where {T}
    vals = Any[]
    for f in entry.keys
        f === :dtype && push!(vals, nameof(T))
        f === :dim && push!(vals, Int(dim))
        f === :nbucket && push!(vals, nbucket(n))
    end
    return Tuple(vals)
end

"""
    resolve_variant(entry, T, dim, n) -> Vector{KernelStage} or `nothing`

The compiled stages for one problem, or `nothing` if this entry has none.

An exact key match wins. Failing that, `:nbucket` alone is allowed to fall back — to the smallest
registered bucket ABOVE the requested one, then the largest below — because a kernel compiled for a
longer sequence still computes a shorter one correctly (it is a compile-time upper bound, not an
assumption), whereas a wrong `dtype` or `dim` is a different ABI and must never be substituted.

`nothing` is a real answer, not an error: it is what triggers a lazy tune for `:default`, and a
clear "compile this configuration first" for anything else.
"""
function resolve_variant(entry::KernelEntry, ::Type{T}, dim::Integer, n::Integer) where {T}
    key = variant_key(entry, T, dim, n)
    haskey(entry.variants, key) && return entry.variants[key]

    idx = findfirst(==(:nbucket), entry.keys)
    idx === nothing && return nothing

    want = key[idx]
    others(k) = (k[1:idx-1]..., k[idx+1:end]...)
    rest = others(key)
    cands = [k for k in keys(entry.variants) if others(k) == rest]
    isempty(cands) && return nothing

    above = filter(k -> k[idx] >= want, cands)
    pick = isempty(above) ? argmax(k -> k[idx], cands) : argmin(k -> k[idx], above)
    return entry.variants[pick]
end

kernel_registry_dir() = joinpath(kernel_cache_root(), "kernels")
entry_dir(name::Symbol) = joinpath(kernel_registry_dir(), String(name))

# The role sets our own flash kernel uses. They are also the defaults for the
# `register_kernel!(; name, fwd, dq, dk, dv)` form, so a kernel written against this package's
# documented convention needs no role declarations at all.
const _DEFAULT_ROLES = Dict(
    :fwd => (outputs=[:o, :lse],
        inputs=[:q, :k, :v, :bias, :mask],
        scalars=[:scale, :neg_inf, :n, :nh, :mheads]),
    # The backward stages take NO `neg_inf`: only the forward needs a sentinel to initialise the
    # running softmax maximum, and the backward reads `lse` instead of recomputing it. Declaring one
    # here would insert a Float32 ahead of `n` and shift every scalar after it, which does not
    # raise, it just makes `n` garbage and every loop a no-op, so the gradients come back all zero.
    :dq => (outputs=[:dq, :dbias],
        inputs=[:q, :k, :v, :bias, :mask, :dO, :lse_in, :delta_in],
        scalars=[:scale, :n, :nh, :mheads]),
    :dk => (outputs=[:dk],
        inputs=[:q, :k, :v, :bias, :mask, :dO, :lse_in, :delta_in],
        scalars=[:scale, :n, :nh, :mheads]),
    :dv => (outputs=[:dv],
        inputs=[:q, :k, :bias, :mask, :dO, :lse_in],
        scalars=[:scale, :n, :nh, :mheads]),
)

const _DEFAULT_LAYOUT = [:hd, :H, :Nq, :Nj, :B]

"""
    _validate_stages(stages)

Check an entry's stages before anything is copied or registered.

Every check here is for a failure that is SILENT at launch: a missing gradient simply stays zero, a
role read before it is written binds uninitialised memory, and an unknown role would be dropped
from the buffer table and shift every slot after it. None of these raise on their own.
"""
function _validate_stages(stages::Vector{KernelStage})
    isempty(stages) && throw(ArgumentError("a kernel entry needs at least a `:fwd` stage"))
    any(s -> s.name === :fwd, stages) ||
        throw(ArgumentError("a kernel entry needs a `:fwd` stage; got " *
                            join(["`:$(s.name)`" for s in stages], ", ")))

    for s in stages
        for r in s.outputs
            r in _OUTPUT_ROLES || throw(ArgumentError(
                "stage `:$(s.name)` declares `:$r` as an output, but it is not an output role"))
        end
        for r in s.inputs
            r in _INPUT_ROLES || throw(ArgumentError(
                "stage `:$(s.name)` declares `:$r` as an input, but it is not an input role"))
        end
        for r in s.scalars
            r in _SCALAR_ROLES || throw(ArgumentError(
                "stage `:$(s.name)` declares `:$r` as a scalar, but it is not a scalar role"))
        end
        all(g -> g in _GRID_AXES, s.grid) || throw(ArgumentError(
            "stage `:$(s.name)` has grid $(s.grid); each axis must be one of " *
            join([":$g" for g in _GRID_AXES], ", ")))
        s.threadgroup > 0 || throw(ArgumentError(
            "stage `:$(s.name)` has threadgroup $(s.threadgroup); must be positive"))
    end

    # Each gradient written exactly once. Twice means one launch silently overwrites the other;
    # zero means that gradient is whatever the freshly-allocated buffer happened to hold.
    bwd = filter(s -> s.name !== :fwd, stages)
    if !isempty(bwd)
        for g in (:dq, :dk, :dv, :dbias)
            n = count(s -> g in s.outputs, bwd)
            n == 1 || throw(ArgumentError(
                "`:$g` is written by $n backward stages; it must be written by exactly one. " *
                "Stages: " * join(["`:$(s.name)` -> $(s.outputs)" for s in bwd], ", ")))
        end
    end

    # A stage may only read a residual that an EARLIER stage produced (or that the runtime supplies
    # itself, which is the case for `o`, `lse` and `delta`). Reading one a LATER stage writes binds
    # a buffer that has not been filled in yet.
    for (i, s) in enumerate(stages)
        for r in s.inputs
            src = get(_INPUT_SOURCE, r, nothing)
            src === nothing && continue
            later = findfirst(j -> src in stages[j].outputs, (i+1):length(stages))
            later === nothing || throw(ArgumentError(
                "stage `:$(s.name)` reads `:$r`, but `:$src` is not written until stage " *
                "`:$(stages[i+later].name)`, which runs later. Reorder the stages."))
        end
    end
    return nothing
end

"""
    _LOADED_BACKENDS

Backends whose extension has loaded AND whose device is usable, as `:metal`/`:cuda`/`:rocm`. Each
backend extension adds itself in `__init__`. [`use_kernel!`](@ref) consults it so selecting a CUDA
kernel on a Metal machine fails at the point of selection rather than at the first launch.
"""
const _LOADED_BACKENDS = Set{Symbol}()

"""
    backend_loaded(backend) -> Bool

Whether `backend`'s extension is loaded and its device usable. See [`_LOADED_BACKENDS`](@ref).
"""
backend_loaded(backend::Symbol) = backend in _LOADED_BACKENDS

_stage_entrypoint(path) = splitext(basename(path))[1]

"""
    _build_stage(name, file, spec; grid, threadgroup) -> KernelStage

One stage from a file plus a role spec, filling in [`_DEFAULT_ROLES`](@ref) for the four stage names
our own kernel uses.
"""
function _build_stage(sname::Symbol, file::AbstractString, spec;
    grid::NTuple{3,Symbol}, threadgroup::Int)
    roles = spec === nothing ? get(_DEFAULT_ROLES, sname) do
        throw(ArgumentError(
            "stage `:$sname` has no default role set; pass it explicitly via `stages=`. " *
            "Defaults exist only for " * join(["`:$k`" for k in keys(_DEFAULT_ROLES)], ", ")))
    end : spec
    return KernelStage(sname, basename(file), _stage_entrypoint(file),
        collect(roles.outputs), collect(roles.inputs), collect(roles.scalars),
        grid, threadgroup)
end

"""
    register_kernel!(; name, fwd, dq, dk, dv, dim, kwargs...) -> KernelEntry

Register a compiled kernel, or add a configuration to one already registered.

`fwd`, `dq`, `dk` and `dv` are paths to compiled binaries — `.metallib`, `.cubin`/`.fatbin`, or
`.hsaco`/`.co`. All must share a format; the backend is inferred from it. The files are COPIED into
the registry cache (`kernel_cache_root()/kernels/<name>/`) under stable names, so the entry keeps
working after the originals move, and a TOML manifest is written beside them — which is what lets
the registry rehydrate on the next Julia start without the user re-registering anything.

# Keyword arguments
- `name`: entry name. `:$(RESERVED_KERNEL_NAME)` is reserved — see [`tune_flash_kernel!`](@ref).
- `dim`: head dimension the binaries were compiled for.
- `dtype`: storage element type they were compiled for. Defaults to `Float32`.
- `n`: sequence length they were compiled for, if any. Pass it only when the binaries bake the
  length in (trifast's `CLOSEST_N`, say); leave it `nothing` when `n` is a runtime argument, and
  the entry then serves every length from one binary. Bucketed by [`nbucket`](@ref).
- `layout`/`rowmajor`: the axis order the binaries expect. Defaults to this package's own
  `$(_DEFAULT_LAYOUT)`, column-major.
- `threadgroup`, `grid`: launch geometry, applied to every stage unless `stages` overrides.
- `stages`: the general form. A vector of NamedTuples `(; name, file, outputs, inputs, scalars,
  grid, threadgroup)`, for a kernel whose backward does not split the way ours does — trifast's
  `dq`(+`delta`) / `dk`+`dv` / `dbias`, for instance. Supersedes `fwd`/`dq`/`dk`/`dv`. `grid` and
  `threadgroup` default to the keyword arguments of the same name if omitted.
- `preamble`: leading `:bytes` slots. Empty unless you know you need them.

# The entry-point name is taken from the file
There is no `entry=` to set. The symbol looked up inside each binary is the file's own basename with
its extension stripped — [`register_kernel!`](@ref)`(; fwd="build/my_fwd.cubin", ...)` looks up
`"my_fwd"` inside `my_fwd.cubin`. Whatever name your AOT compiler gave the kernel at compile time
(`GPUCompiler`'s `name=` keyword, Triton's function name, …), the FILE must be named to match.

# The calling convention
Buffers are bound in a DERIVED order — see [`abi_order`](@ref) and this file's header. A kernel
whose arguments are in a different order needs a shim that reorders them; the registry deliberately
offers no way to describe an arbitrary slot table, because a wrong one binds silently and produces
wrong numbers rather than an error.

# Example
```julia
register_kernel!(; name = :my_cuda,
    fwd = "build/fwd.cubin", dq = "build/dq.cubin",
    dk = "build/dk.cubin",   dv = "build/dv.cubin",
    dim = 32, description = "hand-tuned H100 kernel")
use_kernel!(:my_cuda)
```

See also [`use_kernel!`](@ref), [`unregister_kernel!`](@ref), [`list_kernels`](@ref).
"""
function register_kernel!(; name::Symbol,
    fwd=nothing, dq=nothing, dk=nothing, dv=nothing,
    stages=nothing, dim::Integer, dtype::Type=Float32, n=nothing,
    layout=_DEFAULT_LAYOUT, rowmajor::Bool=false,
    threadgroup::Integer=1024, grid::NTuple{3,Symbol}=(:cdiv_n_bj, :n, :bh),
    preamble=Symbol[], description::AbstractString="")

    name === RESERVED_KERNEL_NAME && throw(ArgumentError(
        "`:$RESERVED_KERNEL_NAME` is reserved for this package's own KernelAbstractions flash " *
        "kernel and is written only by `tune_flash_kernel!`. Choose another name."))

    return _register!(name; fwd, dq, dk, dv, stages, dim, dtype, n, layout, rowmajor,
        threadgroup, grid, preamble, description)
end

# The unreserved half of `register_kernel!`, so `tune_flash_kernel!` can write `:default` without
# being able to reach it from user code.
function _register!(name::Symbol; fwd, dq, dk, dv, stages, dim, dtype, n,
    layout, rowmajor, threadgroup, grid, preamble, description)

    srcs = Pair{Symbol,String}[]
    specs = Dict{Symbol,Any}()
    gridof = Dict{Symbol,NTuple{3,Symbol}}()
    tgof = Dict{Symbol,Int}()

    if stages !== nothing
        for s in stages
            push!(srcs, s.name => String(s.file))
            specs[s.name] = (outputs=s.outputs, inputs=s.inputs, scalars=s.scalars)
            gridof[s.name] = get(s, :grid, grid)
            tgof[s.name] = Int(get(s, :threadgroup, threadgroup))
        end
    else
        for (sname, p) in ((:fwd, fwd), (:dq, dq), (:dk, dk), (:dv, dv))
            p === nothing && continue
            push!(srcs, sname => String(p))
            specs[sname] = nothing
            gridof[sname] = grid
            tgof[sname] = Int(threadgroup)
        end
    end
    isempty(srcs) && throw(ArgumentError("no kernel files given; pass at least `fwd=`"))

    for (_, p) in srcs
        isfile(p) || throw(ArgumentError("kernel file `$p` does not exist"))
    end
    format, backend = _classify(last.(srcs))

    dir = entry_dir(name)
    mkpath(dir)
    built = KernelStage[]
    for (sname, src) in srcs
        ext = lowercase(splitext(src)[2])
        dst = joinpath(dir, "$(name)_$(sname)$(ext)")
        # `force` so re-registering a name genuinely replaces the binary rather than silently
        # keeping the old one, which would be indistinguishable from a successful update.
        cp(src, dst; force=true)
        st = _build_stage(sname, src, specs[sname];
            grid=gridof[sname], threadgroup=tgof[sname])
        push!(built, KernelStage(st.name, basename(dst), st.entry,
            st.outputs, st.inputs, st.scalars, st.grid, st.threadgroup))
    end
    _validate_stages(built)

    kfields = Symbol[:dtype, :dim]
    n === nothing || push!(kfields, :nbucket)

    prev = get(_KERNELS, name, nothing)
    entry = if prev === nothing
        KernelEntry(name, backend, format, collect(Symbol.(layout)), rowmajor,
            kfields, Dict{Tuple,Vector{KernelStage}}(), collect(Symbol.(preamble)),
            String(description))
    else
        prev.backend === backend || throw(ArgumentError(
            "`:$name` is already registered for backend `:$(prev.backend)`; refusing to add a " *
            "`:$backend` variant to it. Unregister it first, or use a different name."))
        prev.keys == kfields || throw(ArgumentError(
            "`:$name` is keyed on $(prev.keys); this registration is keyed on $kfields. " *
            "Every variant of one entry must agree on whether `n` is baked in."))
        prev
    end

    key = variant_key(entry, dtype, dim, something(n, 1))
    entry.variants[key] = built
    _KERNELS[name] = entry
    _save_entry(entry)
    return entry
end

"""
    unregister_kernel!(name) -> Bool

Remove an entry and its cached binaries. Returns whether anything was removed.

If it was the selected kernel, the selection reverts to `:$(RESERVED_KERNEL_NAME)` rather than
being left pointing at nothing — [`LTA_KERNEL`](@ref) is never empty.
"""
function unregister_kernel!(name::Symbol)
    haskey(_KERNELS, name) || return false
    delete!(_KERNELS, name)
    rm(entry_dir(name); recursive=true, force=true)
    current_kernel() === name && (LTA_KERNEL[] = static(RESERVED_KERNEL_NAME))
    return true
end

"""
    list_kernels() -> Vector{NamedTuple}

Every registered entry, with the marker `selected` on the one [`LTA_KERNEL`](@ref) names.

The first thing to check when a kernel is not running: if it is absent, its extension never loaded
or it was never registered; if it is present but not `selected`, call [`use_kernel!`](@ref).
"""
list_kernels() = [(; e.name, e.backend, e.format, nvariants=length(e.variants),
    selected=(e.name === current_kernel()), e.description)
        for e in sort(collect(values(_KERNELS)); by=e -> e.name)]

"""
    use_kernel!(name) -> Symbol

Select `name` as the kernel every `_sdpa_kernel` call will use, and return it.

Checks that the entry exists and that its backend is actually available here — selecting a CUDA
kernel on a Metal machine fails now, with a list of what IS loaded, rather than at the first launch
inside a training loop.

The selection is session-local: it is not persisted, so it must be made again in a new process. The
REGISTRY does persist, so `register_kernel!` need not be.
"""
function use_kernel!(name::Symbol)
    entry = kernel_entry(name)
    if !backend_loaded(entry.backend)
        throw(ArgumentError(
            "`:$name` is a `:$(entry.backend)` kernel, but that backend is not available here " *
            "(loaded: " * (isempty(_LOADED_BACKENDS) ? "none" :
                           join(sort([string(b) for b in _LOADED_BACKENDS]), ", ")) *
            "). Load the matching package, or select a kernel for a backend you have."))
    end
    LTA_KERNEL[] = static(name)
    return name
end

# The registry survives a restart; the SELECTION does not (see `LTA_KERNEL`). TOML rather than
# serialisation so an entry can be read, diffed and hand-edited, the same reasoning as the
# calibration cache, and for the same reason: this is exactly the kind of thing someone will want
# to inspect or pin in CI.

_manifest_path(name::Symbol) = joinpath(entry_dir(name), "$(name).toml")

_stage_dict(s::KernelStage) = Dict{String,Any}(
    "name" => String(s.name), "file" => s.file, "entry" => s.entry,
    "outputs" => String.(s.outputs), "inputs" => String.(s.inputs),
    "scalars" => String.(s.scalars),
    "grid" => [String(g) for g in s.grid], "threadgroup" => s.threadgroup)

_stage_from(d::AbstractDict) = KernelStage(
    Symbol(d["name"]), d["file"], d["entry"],
    Symbol.(d["outputs"]), Symbol.(d["inputs"]), Symbol.(d["scalars"]),
    (Symbol(d["grid"][1]), Symbol(d["grid"][2]), Symbol(d["grid"][3])),
    Int(d["threadgroup"]))

# Key elements are `Symbol` (a dtype name) or `Int`, and TOML arrays are homogeneous, so they go
# out as strings and are parsed back against `entry.keys`, which is what says how to read each
# position. Storing them untyped would make `:dim => 32` and a dtype named "32" indistinguishable.
_key_strings(key::Tuple) = [string(x) for x in key]

function _key_from(strs, kfields::Vector{Symbol})
    length(strs) == length(kfields) || return nothing
    vals = Any[]
    for (f, s) in zip(kfields, strs)
        f === :dtype ? push!(vals, Symbol(s)) : push!(vals, parse(Int, s))
    end
    return Tuple(vals)
end

"""
    _save_entry(entry) -> String

Write `entry`'s manifest beside its binaries. Called on every registration, so the on-disk form is
never behind the in-memory one.
"""
function _save_entry(entry::KernelEntry)
    dir = entry_dir(entry.name)
    mkpath(dir)
    tbl = Dict{String,Any}(
        "name" => String(entry.name),
        "backend" => String(entry.backend),
        "format" => String(entry.format),
        "layout" => String.(entry.layout),
        "rowmajor" => entry.rowmajor,
        "keys" => String.(entry.keys),
        "preamble" => String.(entry.preamble),
        "description" => entry.description,
        "variant" => [Dict{String,Any}(
            "key" => _key_strings(k),
            "stage" => [_stage_dict(s) for s in v])
                      for (k, v) in sort(collect(entry.variants); by=first ∘ string ∘ first)],
    )
    path = _manifest_path(entry.name)
    open(path, "w") do io
        TOML.print(io, tbl)
    end
    return path
end

"""
    _load_entry(path) -> KernelEntry or `nothing`

Read one manifest. Returns `nothing` — never throws — when the file is corrupt or any binary it
names has gone missing.

A half-present cache must not stop the package from loading. The cache directory is under
`~/.cache` where anything may have been pruned, and the correct response to a broken entry is to
ignore it (leaving `:$(RESERVED_KERNEL_NAME)`, which can always be rebuilt) rather than to make
`using LuxTriangleAttention` fail.
"""
function _load_entry(path::AbstractString)
    return try
        raw = TOML.parsefile(path)
        name = Symbol(raw["name"])
        kfields = Symbol.(raw["keys"])
        dir = dirname(path)
        variants = Dict{Tuple,Vector{KernelStage}}()
        for vd in get(raw, "variant", Any[])
            key = _key_from(vd["key"], kfields)
            key === nothing && continue
            sts = [_stage_from(sd) for sd in vd["stage"]]
            # A manifest naming a binary that no longer exists is worse than no manifest: it would
            # be selected, then fail at launch. Drop the variant instead.
            all(s -> isfile(joinpath(dir, s.file)), sts) || continue
            variants[key] = sts
        end
        isempty(variants) ? nothing :
        KernelEntry(name, Symbol(raw["backend"]), Symbol(raw["format"]),
            Symbol.(raw["layout"]), Bool(raw["rowmajor"]), kfields, variants,
            Symbol.(get(raw, "preamble", String[])), get(raw, "description", ""))
    catch
        nothing
    end
end

"""
    load_registry!() -> Vector{Symbol}

Rehydrate every entry under the registry cache, returning the names loaded.

Called from the package's `__init__`, so a kernel registered in a previous session is available
immediately. Never throws: see [`_load_entry`](@ref).
"""
function load_registry!()
    dir = kernel_registry_dir()
    isdir(dir) || return Symbol[]
    loaded = Symbol[]
    for sub in readdir(dir; join=true)
        isdir(sub) || continue
        path = joinpath(sub, basename(sub) * ".toml")
        isfile(path) || continue
        e = _load_entry(path)
        e === nothing && continue
        _KERNELS[e.name] = e
        push!(loaded, e.name)
    end
    return loaded
end

"""
    stage_path(entry, stage) -> String

Absolute path of a stage's binary inside the registry cache.
"""
stage_path(entry::KernelEntry, s::KernelStage) = joinpath(entry_dir(entry.name), s.file)

"""
    find_stage(stages, name) -> KernelStage or `nothing`

The stage called `name`, or `nothing`. Used to pull `:fwd` out of a variant's stage list.
"""
find_stage(stages, name::Symbol) =
    (i = findfirst(s -> s.name === name, stages); i === nothing ? nothing : stages[i])
