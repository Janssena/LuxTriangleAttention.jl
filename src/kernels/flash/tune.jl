# Choosing (TG, KT) for the flash kernel, without any hardcoded device constants.
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
# `tref` is measured once per (device, dim) at an aligned probe size and normalised per unit of
# useful work. Choosing a configuration for a new `n` is then an argmin over a handful of cached
# numbers: no benchmarking per `n`, and no cache keyed by `n`.
#
# An earlier version hardcoded "cap TG at 128, KT=64 for TG=128 else 32, tolerate 1/8 lane waste".
# Every one of those numbers was an M1 Pro measurement promoted to a constant, and none would
# survive a change of vendor. They now EMERGE from `tref` on whatever device is calibrated.
#
# Measured against a realistic distribution (`n = M + [90,150]`), even PINNING one configuration for
# the whole range costs at most 4.8% against per-`n` selection at M=256 and 0.0% at M=512. Since
# per-`n` selection is free, we simply always do it.
#
# What DOES need an ahead-of-time knob is the calibration itself: it compiles and times every
# candidate, so it must not happen inside a latency-sensitive first call. Hence `calibrate` persists
# to disk and `calibrate!` exists for the user to run deliberately.

"""
    DeviceCaps

What a backend must report for the candidate set to be built. Deliberately small: anything that can
be measured is measured rather than declared.

# Fields
- `warp`: threads per warp / wavefront / SIMD group. Group sizes are multiples of this.
- `max_threads`: the device's ceiling on threads per group. A NECESSARY but not sufficient bound —
  see `dispatchable`.
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
    candidate_configs(caps, dim, T; dispatchable) -> Vector{Tuple{Int,Int}}

Legal `(tg, kt)` pairs for a device, head dim and STORAGE type. Independent of `n` by construction.

`T` genuinely changes the space rather than merely scaling it: the staged key/value tiles are held
in storage precision, so halving `sizeof(T)` makes larger `KT` legal. A tuning result is therefore
only valid for the `T` it was measured at.

`dispatchable(tg, kt) -> Bool` is supplied by the backend and is the ONLY reliable filter. A
configuration's real ceiling is the occupancy limit of the **compiled** kernel, which register
pressure drives well below `caps.max_threads`, for this kernel at dim=32 on an M1 Pro, 384 against
a device maximum of 1024. Over-dispatching does not raise: it returns garbage *and* an impossibly
fast time, which is indistinguishable from a tuner finding a huge win unless it is filtered out
here. Every candidate must be compiled and asked.
"""
function candidate_configs(caps::DeviceCaps, dim::Integer, ::Type{T}; dispatchable) where {T}
    out = Tuple{Int,Int}[]
    for tg in caps.warp .* (1, 2, 4, 8, 16, 32)
        tg <= caps.max_threads || continue
        for kt in (16, 32, 64, 128)
            2 * dim * kt * sizeof(T) <= caps.max_shmem || continue
            dispatchable(tg, kt) || continue
            push!(out, (tg, kt))
        end
    end
    return out
end

"""
    lane_waste(n, tg) -> Float64

Lanes launched per lane of useful work. Exact, closed-form, and the only way `n` enters the cost.
"""
lane_waste(n::Integer, tg::Integer) = cld(n, tg) * tg / n

# Keyed by (device, dim, storage type). Leaving `T` out would let a Float16 calibration be served
# for a Float32 request — silently, and with a different legal config set.
const _TREF = Dict{Tuple{String,Int,DataType},Dict{Tuple{Int,Int},Float64}}()

_slug(s::AbstractString) = replace(strip(s), r"[^A-Za-z0-9._-]+" => "_")

"""
    calibration_path(device, dim, T) -> String

Where a calibration is persisted. TOML, so it can be read, diffed and hand-edited, a tuning result
is exactly the kind of thing someone will want to inspect or pin in CI.
"""
calibration_path(device::AbstractString, dim::Integer, ::Type{T}) where {T} =
    joinpath(calibration_cache_dir(),
        "$(_slug(device))__d$(dim)__$(lowercase(string(nameof(T)))).toml")

"""
    save_calibration(path, tref)

Persist measured timings. Tuple keys become `"tg,kt"` strings; non-finite entries (configurations
that failed to build or run) are dropped rather than written as `Inf`, which TOML cannot represent.
"""
function save_calibration(path::AbstractString, tref::AbstractDict)
    mkpath(dirname(path))
    tbl = Dict{String,Any}("timings" => Dict{String,Any}(
        "$(tg),$(kt)" => t for ((tg, kt), t) in tref if isfinite(t)))
    open(path, "w") do io
        TOML.print(io, tbl)
    end
    return path
end

"""
    load_calibration(path) -> Dict or `nothing`

Read a persisted calibration, or `nothing` if it is absent or unreadable. A corrupt cache must never
be fatal — the correct response is to measure again, not to refuse to run.
"""
function load_calibration(path::AbstractString)
    isfile(path) || return nothing
    return try
        raw = TOML.parsefile(path)
        tref = Dict{Tuple{Int,Int},Float64}()
        for (k, v) in get(raw, "timings", Dict{String,Any}())
            tg, kt = parse.(Int, split(k, ","))
            tref[(tg, kt)] = Float64(v)
        end
        isempty(tref) ? nothing : tref
    catch
        nothing
    end
end

"""
    calibrate(caps, dim, T; time_config, dispatchable, probe_n=512, force=false) -> Dict

Measure this device once and cache a per-configuration reference time.

`time_config(tg, kt, n) -> seconds` runs the kernel; the backend owns it because only the backend
knows how to launch. Results are cached per `(device, dim, T)`.

A configuration that throws is recorded as `Inf` rather than aborting the sweep: a device that
refuses one tile size should not prevent the rest from being usable.
"""
function calibrate(caps::DeviceCaps, dim::Integer, ::Type{T}; time_config, dispatchable,
    probe_n::Integer=512, force::Bool=false, persist::Bool=true) where {T}
    key = (caps.name, Int(dim), T)
    (!force && haskey(_TREF, key)) && return _TREF[key]

    # Disk before measurement: the whole point of persisting is that a fresh process is a file read
    # rather than sixteen compile-and-time cycles.
    path = calibration_path(caps.name, dim, T)
    if !force
        cached = load_calibration(path)
        if cached !== nothing
            _TREF[key] = cached
            return cached
        end
    end

    tref = Dict{Tuple{Int,Int},Float64}()
    for (tg, kt) in candidate_configs(caps, dim, T; dispatchable)
        t = try
            time_config(tg, kt, probe_n)
        catch
            Inf
        end
        # Divide out lane waste at the probe size so the entry is per unit of USEFUL work and is
        # comparable across configurations with different TG.
        tref[(tg, kt)] = t / lane_waste(probe_n, tg)
    end
    isempty(tref) &&
        error("no dispatchable configuration for dim=$dim, T=$T on $(caps.name)")
    _TREF[key] = tref
    persist && save_calibration(path, tref)
    return tref
end

"""
    choose_config(tref, n) -> (tg, kt)

The fastest `(tg, kt)` configuration for sequence length `n`, from a calibration produced by
[`calibrate!`](@ref). No benchmarking or allocation — an argmin over cached numbers.
"""
function choose_config(tref::AbstractDict, n::Integer)
    best = nothing
    bestcost = Inf
    for ((tg, kt), t) in tref
        c = t * lane_waste(n, tg)
        if c < bestcost
            bestcost = c
            best = (tg, kt)
        end
    end
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
    tune_flash_kernel!(n; dim, T=Float32, force=false) -> KernelEntry

Tune, compile and register this package's own flash kernel for sequence length `n`.

This is the only writer of the reserved `:$(RESERVED_KERNEL_NAME)` entry, [`register_kernel!`](@ref)
refuses that name. It:

 1. calibrates the device for `(dim, T)`, or reuses the persisted calibration ([`calibrate!`](@ref));
 2. picks the `(tg, kt)` configuration that minimises cost AT `n`, via [`choose_config`](@ref);
 3. ahead-of-time compiles the forward and all three backward kernels for it;
 4. registers them as a variant keyed `(dtype, dim, nbucket(n))`, under file names carrying that
    variant's [`stage_id`](@ref) — so tuning `dim=16` and then `dim=32` leaves BOTH tuned kernels on
    disk and each `(dtype, dim, n)` resolves to its own binary.

# Why `n` is a key rather than a per-call choice
The compiled BINARY is independent of `n`, the kernel takes the length as a runtime argument. What
depends on `n` is the tuned `(tg, kt)`, and the best choice genuinely moves with `n` (by several x
between the best and worst legal configuration). Keying the variant on [`nbucket`](@ref) means
`tune_flash_kernel!` can be called for each length that matters and each call keeps its own tuning,
rather than the last call silently retuning the others. Calls at a length with no exact variant fall
back to the nearest bucket above.

Costs seconds to tens of seconds on a cold cache, because step 1 compiles and times every candidate
configuration. Run it deliberately, ahead of latency-sensitive work.

Requires a backend extension; throws if none is loaded.
"""
function tune_flash_kernel!(n::Integer; dim::Integer, T::Type=Float32, force::Bool=false)
    device_kernel_available() || throw(ErrorException(
        "no usable device backend is loaded, so the built-in flash kernel cannot be compiled. " *
        "Load the package for your GPU (e.g. `using Metal`) first."))

    tref = calibrate!(T; dim, force)
    tg, kt = choose_config(tref, n)

    fwd_path, _ = device_binary(T, dim, tg, kt)
    bwd = device_binary_bwd(T, dim, tg, kt)

    entry = _register!(RESERVED_KERNEL_NAME;
        fwd=fwd_path, dq=bwd.dq[1], dk=bwd.dk[1], dv=bwd.dv[1],
        stages=nothing, dim, dtype=T, n=Int(n),
        layout=_DEFAULT_LAYOUT, rowmajor=false,
        threadgroup=tg, grid=FLASH_GRID,
        preamble=FLASH_PREAMBLE,
        description="built-in KernelAbstractions flash kernel (tuned per sequence length)")

    @debug "tuned flash kernel" n dim T tg kt nbucket = nbucket(n) ids = [s.id for s in
        entry.variants[variant_key(entry, T, dim, n)]]
    return entry
end

"""
    flash_config(entry, T, dim, n) -> (tg, kt)

Recover the `(tg, kt)` a registered `:$(RESERVED_KERNEL_NAME)` variant was compiled for.

The threadgroup size is on the stage; `kt` is not, so it is re-derived from the calibration exactly
as [`tune_flash_kernel!`](@ref) chose it. Both come from the same `choose_config` call on the same
persisted numbers, so this agrees with what was compiled as long as the calibration has not been
cleared — and if it has, the argmin is recomputed over fresh measurements of the same device.
"""
function flash_config(::Type{T}, dim::Integer, n::Integer) where {T}
    tref = calibrate!(T; dim)
    return choose_config(tref, n)
end
