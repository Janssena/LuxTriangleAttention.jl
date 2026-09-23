# Choosing each stage's launch configuration for the flash kernel, without any hardcoded device
# constants.
#
# ## What is tuned
#
# Every stage — `:fwd`, `:dq`, `:dk`, `:dv` — gets its OWN [`FlashConfig`](@ref): the group size `TG`,
# the key-tile size `KT`, and for the forward the softmax sub-block `SB`. They differ because the
# kernels' register footprints differ (`2*DIM` floats per thread for `:fwd`/`:dv`, `3*DIM` for
# `:dq`/`:dk`, see `backward.jl`), and a device's register ceiling is not something to declare: it is
# measured, as the thing that makes an over-full configuration slow or undispatchable. `SB` is the
# forward's one knob that trades registers for arithmetic (one softmax rescale per `SB` keys instead
# of per key); on an M1 Pro `SB = 4` wins and `SB = 8` spills, and a device with a larger register
# file can be expected to want more.
#
# ## Why there is no `n` in the calibration
#
# Once the divisibility constraint is gone, `n` enters the cost in exactly ONE closed-form way: a
# group covers `TG` queries, so `cld(n,TG)*TG / n` lanes are launched per lane of useful work.
# Everything else about a configuration is a property of the device. Hence
#
#     cost(cfg, n) = tref[cfg] * cld(n,TG)*TG/n
#                    ^measured once  ^exact, free
#
# `tref` is measured once per (device, dim, T) and stage at an aligned probe size and normalised per
# unit of useful work. Choosing a configuration for a new `n` is then an argmin over cached numbers:
# no benchmarking per `n`, and no cache keyed by `n`.
#
# ## How the search stays affordable
#
# The full space is `TG × KT × SB` for the forward — about a hundred configurations — and every one
# must be compiled, checked for dispatchability and correctness, and timed. Instead the forward is
# searched by coordinate descent: `SB` at a starting geometry, then `(TG, KT)` at the best `SB`, then
# `SB` again at the best geometry. `(TG, KT)` and `SB` interact mainly through the register budget,
# which is exactly what the final pass re-checks. The backward stages have no `SB` and are swept in
# full. A calibration is still seconds to minutes, which is why it is persisted and run deliberately
# (`calibrate!`), never inside a latency-sensitive first call.
#
# ## Every candidate is checked before its time counts
#
# See `verify.jl`. Over-dispatching a pipeline past its occupancy limit returns garbage AND an
# impossibly fast time; a miscompiled tile shape is silently wrong. Either would read as a tuner
# finding a huge win. A candidate that fails dispatchability or verification is recorded as `Inf` and
# can never be chosen.

"""
    DeviceCaps

What a backend must report for the candidate set to be built. Deliberately small: anything that can
be measured is measured rather than declared.

# Fields
- `warp`: threads per warp / wavefront / SIMD group. Group sizes are multiples of this.
- `max_threads`: the device's ceiling on threads per group. A NECESSARY but not sufficient bound —
  the compiled pipeline's own limit is what `dispatchable` checks.
- `max_shmem`: bytes of local memory available to one group.
- `name`: device identifier, used as the calibration cache key.
"""
struct DeviceCaps
    warp::Int
    max_threads::Int
    max_shmem::Int
    name::String
end

"""
    FLASH_STAGES

The stages of the built-in kernel, in launch order: the forward, then the three backward passes.
"""
const FLASH_STAGES = (:fwd, :dq, :dk, :dv)

"""
    FlashConfig(tg, kt, sb)

One stage's launch configuration: `tg` threads per group, `kt` keys per staged tile, `sb` keys per
softmax sub-block. `sb` is `1` for the backward stages, which carry no running maximum.
"""
struct FlashConfig
    tg::Int
    kt::Int
    sb::Int
end

Base.string(c::FlashConfig) = "$(c.tg),$(c.kt),$(c.sb)"

function _parse_config(s::AbstractString)
    parts = parse.(Int, split(s, ","))
    length(parts) == 3 || throw(ArgumentError("malformed configuration `$s`"))
    return FlashConfig(parts...)
end

"""
    stage_shmem(stage, dim, T, cfg) -> Int

Local memory, in bytes, one group of `stage` uses at `cfg`: two staged `[dim, kt]` tiles in the
storage type plus the per-key `Float32` vectors (the forward's and `:dq`'s staged mask, `:dk`'s
`lse`+`delta`, `:dv`'s `lse`).
"""
function stage_shmem(stage::Symbol, dim::Integer, ::Type{T}, c::FlashConfig) where {T}
    tiles = 2 * dim * c.kt * sizeof(T)
    stage === :dk && return tiles + 2 * 4 * c.kt
    return tiles + 4 * c.kt
end

"""
    candidate_configs(caps, dim, T, stage) -> Vector{FlashConfig}

Legal configurations for one stage on a device, head dim and STORAGE type. Independent of `n` by
construction.

Legality here is only what can be decided without compiling: group sizes that are multiples of the
SIMD width up to the device maximum, tiles that fit in local memory, and `sb` dividing `kt` (the
forward walks each tile in whole sub-blocks). Whether a configuration's COMPILED pipeline admits
`tg` threads — register pressure drives that well below `caps.max_threads`, for the forward at
dim=32 on an M1 Pro, 384 against a device maximum of 1024 — is the backend's `dispatchable`, asked
during the search, since answering it means compiling.

`T` genuinely changes the space rather than merely scaling it: the staged tiles are held in storage
precision, so halving `sizeof(T)` makes larger `kt` legal. A tuning result is therefore only valid
for the `T` it was measured at.
"""
function candidate_configs(caps::DeviceCaps, dim::Integer, ::Type{T}, stage::Symbol) where {T}
    stage in FLASH_STAGES || throw(ArgumentError("unknown stage `:$stage`"))
    sbs = stage === :fwd ? (1, 2, 4, 8) : (1,)
    out = FlashConfig[]
    for tg in caps.warp .* (1, 2, 4, 8, 16, 32)
        tg <= caps.max_threads || continue
        for kt in (16, 32, 64, 128), sb in sbs
            (sb <= kt && kt % sb == 0) || continue
            c = FlashConfig(tg, kt, sb)
            stage_shmem(stage, dim, T, c) <= caps.max_shmem || continue
            push!(out, c)
        end
    end
    return out
end

"""
    lane_waste(n, tg) -> Float64

Lanes launched per lane of useful work. Exact, closed-form, and the only way `n` enters the cost.
"""
lane_waste(n::Integer, tg::Integer) = cld(n, tg) * tg / n

"""
    search_configs(cands, evaluate) -> Dict{FlashConfig,Float64}

Evaluate candidates by coordinate descent over `sb` and `(tg, kt)` (see this file's header), and
return every configuration evaluated with its cost. `evaluate(cfg) -> Float64` is `Inf` for a
configuration that cannot run or is wrong. With a single `sb` value this is an exhaustive sweep.

The starting geometry is the middle of the candidate list, not a device constant: it only has to be
dispatchable often enough to rank the `sb` values, and the final pass re-ranks them at whatever
geometry actually won.
"""
function search_configs(cands::Vector{FlashConfig}, evaluate)
    isempty(cands) && return Dict{FlashConfig,Float64}()
    seen = Dict{FlashConfig,Float64}()
    ev(c) = get!(() -> Float64(evaluate(c)), seen, c)
    sbs = sort(unique(c.sb for c in cands))
    if length(sbs) == 1
        foreach(ev, cands)
        return seen
    end
    geoms = unique((c.tg, c.kt) for c in cands)
    has(g, sb) = FlashConfig(g..., sb) in cands
    best_sb(g) = argmin(sb -> ev(FlashConfig(g..., sb)), filter(sb -> has(g, sb), sbs))

    g0 = geoms[cld(length(geoms), 2)]
    sb1 = best_sb(g0)
    g1 = argmin(g -> has(g, sb1) ? ev(FlashConfig(g..., sb1)) : Inf, geoms)
    best_sb(g1)
    return seen
end

# Bumped whenever what a calibration MEANS changes, so an old file is re-measured rather than
# misread. Version 1 was a single `(tg, kt)` table for every stage.
const CALIBRATION_VERSION = 2

# Keyed by (device, dim, storage type, geometry). Leaving `T` out would let a Float16 calibration be
# served for a Float32 request — silently, and with a different legal config set. `geometry` is
# `nothing` for the triangle-shaped default calibration, or `(obucket, hbucket)` for one measured at
# a layer's own shape (`tune_flash_kernel!(n; nouter, heads)`).
const _TREF = Dict{Any,Dict{Symbol,Dict{FlashConfig,Float64}}}()

_slug(s::AbstractString) = replace(strip(s), r"[^A-Za-z0-9._-]+" => "_")

"""
    calibration_path(device, dim, T) -> String

Where a calibration is persisted. TOML, so it can be read, diffed and hand-edited, a tuning result
is exactly the kind of thing someone will want to inspect or pin in CI.
"""
calibration_path(device::AbstractString, dim::Integer, ::Type{T}, geometry=nothing) where {T} =
    joinpath(calibration_cache_dir(),
        "$(_slug(device))__d$(dim)__$(lowercase(string(nameof(T))))" *
        (geometry === nothing ? "" : "__o$(geometry[1])_h$(geometry[2])") * ".toml")

"""
    save_calibration(path, tref)

Persist measured timings, one table per stage, keyed `"tg,kt,sb"`. Non-finite entries
(configurations that failed to build, dispatch or verify) are dropped rather than written as `Inf`,
which TOML cannot represent.
"""
function save_calibration(path::AbstractString, tref::AbstractDict)
    mkpath(dirname(path))
    tbl = Dict{String,Any}("version" => CALIBRATION_VERSION)
    for (stage, cfgs) in tref
        tbl[String(stage)] = Dict{String,Any}(string(c) => t for (c, t) in cfgs if isfinite(t))
    end
    open(path, "w") do io
        TOML.print(io, tbl)
    end
    return path
end

"""
    load_calibration(path) -> Dict or `nothing`

Read a persisted calibration, or `nothing` if it is absent, unreadable, from another
[`CALIBRATION_VERSION`](@ref), or missing a stage. A corrupt or stale cache must never be fatal —
the correct response is to measure again, not to refuse to run.
"""
function load_calibration(path::AbstractString)
    isfile(path) || return nothing
    return try
        raw = TOML.parsefile(path)
        get(raw, "version", 1) == CALIBRATION_VERSION || return nothing
        tref = Dict{Symbol,Dict{FlashConfig,Float64}}()
        for stage in FLASH_STAGES
            tbl = get(raw, String(stage), nothing)
            tbl === nothing && return nothing
            cfgs = Dict{FlashConfig,Float64}(_parse_config(k) => Float64(v) for (k, v) in tbl)
            isempty(cfgs) && return nothing
            tref[stage] = cfgs
        end
        tref
    catch
        nothing
    end
end

"""
    calibrate(caps, dim, T; dispatchable, verify, time_config, probe_n=512, force=false,
        persist=true) -> Dict{Symbol,Dict{FlashConfig,Float64}}

Measure this device once per stage and cache a per-configuration reference time.

The backend supplies the three things only it can do, each as `f(stage, cfg)`:
- `dispatchable(stage, cfg) -> Bool`: the configuration compiles and its pipeline admits `cfg.tg`
  threads;
- `verify(stage, cfg) -> Bool`: it reproduces the reference on the problem in `verify.jl`;
- `time_config(stage, cfg, n) -> seconds`: one launch at length `n`.

They are asked in that order and a candidate that fails either check is recorded as `Inf`, never
timed. A configuration that THROWS is recorded as `Inf` too, rather than aborting the sweep: a device
that refuses one tile size should not prevent the rest from being usable. A stage with no finite
candidate is an error — there is then no kernel to run.
"""
function calibrate(caps::DeviceCaps, dim::Integer, ::Type{T}; dispatchable, verify, time_config,
    probe_n::Integer=512, force::Bool=false, persist::Bool=true, geometry=nothing) where {T}
    key = (caps.name, Int(dim), T, geometry)
    (!force && haskey(_TREF, key)) && return _TREF[key]

    # Disk before measurement: the whole point of persisting is that a fresh process is a file read
    # rather than a full search.
    path = calibration_path(caps.name, dim, T, geometry)
    if !force
        cached = load_calibration(path)
        if cached !== nothing
            _TREF[key] = cached
            return cached
        end
    end

    tref = Dict{Symbol,Dict{FlashConfig,Float64}}()
    for stage in FLASH_STAGES
        function evaluate(c::FlashConfig)
            try
                dispatchable(stage, c) || return Inf
                if !verify(stage, c)
                    @warn "flash kernel configuration failed verification; excluded" stage c
                    return Inf
                end
                # Divide out lane waste at the probe size so the entry is per unit of USEFUL work
                # and is comparable across configurations with different TG.
                return time_config(stage, c, probe_n) / lane_waste(probe_n, c.tg)
            catch err
                @debug "flash kernel configuration failed" stage c exception = err
                return Inf
            end
        end
        tref[stage] = search_configs(candidate_configs(caps, dim, T, stage), evaluate)
        any(isfinite, values(tref[stage])) || error(
            "no dispatchable, verified configuration for stage `:$stage`, dim=$dim, T=$T on " *
            "$(caps.name)")
    end
    _TREF[key] = tref
    persist && save_calibration(path, tref)
    return tref
end

"""
    choose_config(tref, n) -> FlashConfig

The fastest configuration of ONE stage for sequence length `n`, from that stage's calibration
table (`calibrate!(T; dim)[stage]`). No benchmarking or allocation — an argmin over cached numbers.
"""
function choose_config(tref::AbstractDict{FlashConfig}, n::Integer)
    best = nothing
    bestcost = Inf
    for (c, t) in tref
        cost = t * lane_waste(n, c.tg)
        if cost < bestcost
            bestcost = cost
            best = c
        end
    end
    best === nothing && error("no usable configuration in this calibration")
    return best
end

"""
    clear_calibration!(; disk=false)

Drop cached measurements. In-memory only by default; `disk=true` also removes the persisted files,
which is what you want after changing the kernel itself, since a stale calibration would then be
measuring something that no longer exists.
"""
function clear_calibration!(; disk::Bool=false)
    empty!(_TREF)
    if disk
        dir = calibration_cache_dir()
        isdir(dir) && for f in readdir(dir; join=true)
            endswith(f, ".toml") && rm(f; force=true)
        end
    end
    return nothing
end

"""
    FLASH_PREAMBLE

The two leading buffer slots our KernelAbstractions kernel needs and no other kernel does: GPUCompiler's
kernel-state slot, and KernelAbstractions' launch context.

They are declared as ROLES rather than as baked bytes because the context is not a constant — it
carries the iterspace, so it depends on `(n, heads, tg)` and is rebuilt at every launch. The
launcher materialises both; the registry only records that this entry expects them, and where.
"""
const FLASH_PREAMBLE = [:kernel_state, :ka_context]

"""
    FLASH_GRID

Launch geometry for every stage of our kernel: a 1-D grid of all groups.

Not a 3-D grid, and that is not a style choice — KernelAbstractions' Metal backend reads only the x
component of the threadgroup position and rebuilds N-D indices in software, so y and z are silently
dropped. See the note at the top of `flash/forward.jl`; the same is likely to apply on any backend.
"""
const FLASH_GRID = (:ka_ngroups, :one, :one)

"""
    FLASH_ROLES

The role sets of the built-in kernel's stages. They differ from the user-facing defaults
(`_DEFAULT_ROLES` in `registry.jl`) in one respect: the forward and `:dq` read the pair bias
TRANSPOSED (`:bias_t`) and `:dq` writes its gradient transposed (`:dbias_t`), because a thread per
QUERY reads a queries-fastest bias coalesced. `:dk` and `:dv`, a thread per KEY, keep `:bias`. The
launchers materialise `:bias_t` and transpose `:dbias_t` back; see `backward.jl`'s header.
"""
const FLASH_ROLES = Dict(
    :fwd => (outputs=[:o, :lse],
        inputs=[:q, :k, :v, :bias_t, :mask],
        scalars=[:scale, :neg_inf, :n, :nouter, :nh, :mheads]),
    :dq => (outputs=[:dq, :dbias_t],
        inputs=[:q, :k, :v, :bias_t, :mask, :dO, :lse_in, :delta_in],
        scalars=[:scale, :n, :nouter, :nh, :mheads]),
    :dk => (outputs=[:dk],
        inputs=[:q, :k, :v, :bias, :mask, :dO, :lse_in, :delta_in],
        scalars=[:scale, :n, :nouter, :nh, :mheads]),
    :dv => (outputs=[:dv],
        inputs=[:q, :k, :bias, :mask, :dO, :lse_in],
        scalars=[:scale, :n, :nouter, :nh, :mheads]),
)

"""
    tune_flash_kernel!(n; dim, T=Float32, force=false, nouter=n, heads=nothing) -> KernelEntry

Tune, compile and register this package's own flash kernel for sequence length `n`, optionally for
one attention GEOMETRY.

# Per-layer tuning: `nouter` and `heads`
The kernel runs one thread per query, `n · nouter · heads` threads in all (`heads` counting every
`(head, batch)` slice), so the best configuration depends on the whole shape, not only on `n`.
Triangle attention (`nouter = n`), MSA row attention (`nouter = N_seq`) and a token sequence
(`nouter = 1`, 16 heads) sit in very different occupancy regimes.

- Without `heads`: the device is calibrated once per `(dim, T)` at triangle geometry and the variant
  is keyed `hbucket = 0`, a wildcard that serves every head count without an exact variant. This is
  what a lazy first eager call does.
- With `heads` (and `nouter`): the calibration is measured AT that geometry (persisted separately),
  and the variant is keyed `(obucket, hbucket) = (nbucket(nouter), nbucket(heads))`. A call with that
  shape then resolves to it; see [`resolve_variant`](@ref) for the fallbacks.

This is the only writer of the reserved `:$(RESERVED_KERNEL_NAME)` entry, [`register_kernel!`](@ref)
refuses that name. It:

 1. calibrates the device for `(dim, T)`, or reuses the persisted calibration ([`calibrate!`](@ref));
 2. picks, PER STAGE, the configuration that minimises cost AT `n`, via [`choose_config`](@ref);
 3. ahead-of-time compiles the forward and all three backward kernels, each at its own
    configuration;
 4. registers them as a variant keyed `(dtype, dim, nbucket(n), nbucket(nouter), hbucket)`, under
    file names carrying that variant's [`stage_id`](@ref) — so tuning `dim=16` and then `dim=32`, or
    two geometries, leaves every tuned kernel on disk and each resolves to its own binary;
 5. builds the no-bias / no-mask specialisations of every stage at the same configurations (see
    [`specialise_stages`](@ref)), so a traced call without a bias or mask never has to compile.

# Why `n` is a key rather than a per-call choice
The compiled BINARY is independent of `n`, the kernel takes the length as a runtime argument. What
depends on `n` is the tuned configuration, and the best choice genuinely moves with `n` (by several x
between the best and worst legal configuration). Keying the variant on [`nbucket`](@ref) means
`tune_flash_kernel!` can be called for each length that matters and each call keeps its own tuning,
rather than the last call silently retuning the others. Calls at a length with no exact variant fall
back to the nearest bucket above.

Costs seconds to minutes on a cold cache, because step 1 compiles, verifies and times candidate
configurations for all four stages. Run it deliberately, ahead of latency-sensitive work.

Requires a backend extension; throws if none is loaded.
"""
function tune_flash_kernel!(n::Integer; dim::Integer, T::Type=Float32, force::Bool=false,
    nouter::Integer=n, heads::Union{Nothing,Integer}=nothing)
    device_kernel_available() || throw(ErrorException(
        "no usable device backend is loaded, so the built-in flash kernel cannot be compiled. " *
        "Load the package for your GPU (e.g. `using Metal`) first."))

    tref = heads === nothing ? calibrate!(T; dim, force) :
        calibrate!(T; dim, force, n=Int(n), nouter=Int(nouter), heads=Int(heads))
    stages = map(collect(FLASH_STAGES)) do s
        cfg = choose_config(tref[s], n)
        path, _ = device_binary(s, T, dim, cfg)
        (; name=s, file=path, FLASH_ROLES[s]..., grid=FLASH_GRID, threadgroup=cfg.tg)
    end

    entry = _register!(RESERVED_KERNEL_NAME;
        fwd=nothing, dq=nothing, dk=nothing, dv=nothing,
        stages, dim, dtype=T, n=Int(n), nouter=Int(nouter), heads=Int(something(heads, 0)),
        layout=_DEFAULT_LAYOUT, rowmajor=false,
        threadgroup=first(stages).threadgroup, grid=FLASH_GRID,
        preamble=FLASH_PREAMBLE,
        description="built-in KernelAbstractions flash kernel (tuned per sequence length)")

    registered = resolve_variant(entry, T, dim, n; nouter, heads=something(heads, 0))
    for (hb, hm) in ((false, true), (true, false), (false, false))
        specialise_stages(entry, registered, T, dim, hb, hm)
    end

    @debug "tuned flash kernel" n dim T configs = Dict(s => choose_config(tref[s], n) for s in
        FLASH_STAGES) nbucket = nbucket(n)
    return entry
end

"""
    flash_config(T, dim, n) -> Dict{Symbol,FlashConfig}

The configuration each stage of a `:$(RESERVED_KERNEL_NAME)` variant for `n` was (or would be)
compiled with, re-derived from the calibration exactly as [`tune_flash_kernel!`](@ref) chose it.
Both come from the same `choose_config` call on the same persisted numbers, so this agrees with what
was compiled as long as the calibration has not been cleared — and if it has, the argmin is
recomputed over fresh measurements of the same device.
"""
function flash_config(::Type{T}, dim::Integer, n::Integer; nouter::Integer=n,
    heads::Union{Nothing,Integer}=nothing) where {T}
    tref = heads === nothing ? calibrate!(T; dim) :
        calibrate!(T; dim, n=Int(n), nouter=Int(nouter), heads=Int(heads))
    return Dict(s => choose_config(tref[s], n) for s in FLASH_STAGES)
end

# ---------------------------------------------------------------------------------------------
# Absent bias / mask: kernel specialisations
# ---------------------------------------------------------------------------------------------

"""
    kernel_skips_absent(entry) -> Bool

Whether `entry` has bias-free / mask-free specialisations, i.e. whether a launcher may bind a
1-element placeholder for an absent bias or mask instead of materialising a zero bias and an
all-live mask. True for the built-in `:$(RESERVED_KERNEL_NAME)` kernel only: a registered third-party
kernel reads whatever it is given, so it keeps receiving full-size buffers.
"""
kernel_skips_absent(entry::KernelEntry) = entry.name === RESERVED_KERNEL_NAME

# `(tg, kt[, sb])` as `build` writes them into the entry-point name, `..._t128_k32_s4_<tag>`.
const _STAGE_CONFIG_RE = r"_t(\d+)_k(\d+)(?:_s(\d+))?_[^_]+$"

"""
    stage_config(stage) -> FlashConfig or `nothing`

The configuration a built-in stage was compiled with, read back from its entry-point name, which
the backend's `build` composes from it. Reading it back rather than re-deriving it from the
calibration guarantees the specialisation uses exactly the tuned configuration of the stage it
replaces — and so finds the binary [`tune_flash_kernel!`](@ref) already built.
"""
function stage_config(s::KernelStage)
    m = match(_STAGE_CONFIG_RE, s.entry)
    m === nothing && return nothing
    return FlashConfig(parse(Int, m[1]), parse(Int, m[2]), m[3] === nothing ? 1 : parse(Int, m[3]))
end

const _SPECIALISED = Dict{Tuple,Vector{KernelStage}}()

"""
    specialise_stages(entry, stages, T, dim, hasbias, hasmask) -> Vector{KernelStage}

The stages to launch for a call whose bias / mask is present (`true`) or `nothing` (`false`).

For the built-in kernel, an absent bias or mask swaps every stage for its specialisation (the `HB`/`HM`
flags of [`flash_fwd!`](@ref)), compiled at the same configuration: same roles, same buffer table,
same grid, a binary that simply never reads the absent operand. Everything else — a call with both
present, or a third-party entry (see [`kernel_skips_absent`](@ref)) — gets `stages` unchanged.

The specialised stage's `file` is the absolute path of the backend's cached binary, which
`stage_path` returns as-is. Memoised per `(stages, flags)`, so the lookup is a dictionary hit after
the first call; the binaries themselves are built by `tune_flash_kernel!`, so a traced call does
not compile.
"""
function specialise_stages(entry::KernelEntry, stages::Vector{KernelStage}, ::Type{T},
    dim::Integer, hasbias::Bool, hasmask::Bool) where {T}
    (hasbias && hasmask) && return stages
    kernel_skips_absent(entry) || return stages
    key = (Tuple(s.entry for s in stages), T, Int(dim), hasbias, hasmask)
    return get!(_SPECIALISED, key) do
        suffix = (hasbias ? "" : "nb") * (hasmask ? "" : "nm")
        map(stages) do s
            cfg = stage_config(s)
            cfg === nothing && error("cannot read the configuration of built-in stage " *
                "`$(s.name)` from its entry name `$(s.entry)`; re-run `tune_flash_kernel!`")
            path, ename = device_binary(s.name, T, dim, cfg; hasbias, hasmask)
            KernelStage(s.name, path, ename, s.id * suffix, s.outputs, s.inputs, s.scalars,
                s.grid, s.threadgroup)
        end
    end
end
