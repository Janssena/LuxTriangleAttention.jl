# ONE tuning skeleton for every built-in kernel. A kernel declares only WHAT is tuned (in its own
# `tune.jl`):
#
#   tune_stages(k)                              the stages, in launch order
#   tune_grid(k, stage, dim, T, caps)           its tunable parameters and their values (a NamedTuple
#                                               of axes; a parameter point is a NamedTuple of values)
#   admissible(k, stage, dim, T, caps, p)       what can be decided without compiling (local memory,
#                                               the device's thread cap, divisibility)
#
# plus how a point becomes a binary (`launch.jl`). Everything else is shared, here.
#
# ## Tuned per sequence length
#
# A configuration's speed depends on the length `n` (a group of `rows` queries leaves part of the
# last group idle unless `rows` divides `n`, and how many groups keep the device busy changes with
# `n`), so each length is tuned AT that length: candidates are timed on a probe of exactly `n`
# queries and the fastest wins. Which lengths are tuned is the user's choice ([`tune_lengths`](@ref),
# default 256 and 512); a call of any other length runs the variant tuned for the closest one.
# Candidates are compiled and verified once per (kernel, dim, T) and stage, and only re-timed for
# each further length.
#
# ## Every candidate is checked before its time counts
#
# Over-dispatching a pipeline past its occupancy limit returns garbage AND an impossibly fast time; a
# miscompiled tile shape is silently wrong. Either would read as the tuner finding a huge win. So a
# candidate must (1) be admitted by its compiled pipeline (`KernelDispatch.max_threads`) and (2) reproduce
# the plain-loop reference on `verification_problem` (`src/kernels/verify.jl`) before it is timed; otherwise, or if
# anything throws, it is recorded as `Inf` and can never be chosen.
#
# ## The search
#
# A grid of at most `FULL_SWEEP_MAX` points is swept in full. A larger one (`:default`'s forward:
# group size × key tile × softmax sub-block, ~100 points, each compiled, verified and timed) is searched
# by coordinate descent over its axes, from the middle of the admissible points: one axis at a time,
# holding the others, until no axis improves. A calibration is still seconds to minutes, which is why
# it is persisted and run deliberately, never inside a latency-sensitive call.

"""The largest grid swept in full; larger ones are searched by coordinate descent."""
const FULL_SWEEP_MAX = 32

"""
    search(points, evaluate) -> Dict{NamedTuple,Float64}

Evaluate parameter points (all with the same axes) and return every point evaluated with its cost.
`evaluate(p) -> Float64` is `Inf` for a point that cannot run or is wrong. At most
[`FULL_SWEEP_MAX`](@ref) points: all of them; more: coordinate descent over the axes from the middle
point (see this file's header), falling back to a full sweep if nothing it reaches is finite.
"""
function search(points::AbstractVector{<:NamedTuple}, evaluate)
    seen = Dict{NamedTuple,Float64}()
    isempty(points) && return seen
    ev(p) = get!(() -> Float64(evaluate(p)), seen, p)
    if length(points) <= FULL_SWEEP_MAX
        foreach(ev, points)
        return seen
    end
    axes_ = keys(first(points))
    cur = points[cld(length(points), 2)]
    improved = true
    while improved
        improved = false
        for ax in axes_
            line = filter(p -> all(a -> a === ax || p[a] == cur[a], axes_), points)
            best = argmin(ev, line)
            if ev(best) < ev(cur)
                cur, improved = best, true
            end
        end
    end
    any(isfinite, values(seen)) || foreach(ev, points)
    return seen
end

"""
    grid_points(kernel, stage, dim, T, caps) -> Vector{NamedTuple}

The admissible points of a stage's [`tune_grid`](@ref), in grid order (first axis fastest).
"""
function grid_points(k::BuiltinKernel, stage::Symbol, dim::Integer, ::Type{T}, caps::DeviceCaps) where {T}
    grid = tune_grid(k, stage, dim, T, caps)
    pts = [NamedTuple{keys(grid)}(t) for t in Iterators.product(values(grid)...)]
    return filter(p -> admissible(k, stage, dim, T, caps, p), vec(pts))
end

# One stage's calibration at one length: each point evaluated, with its time per launch (`Inf`: it
# failed to build, dispatch or verify).
const Calibration = Dict{Symbol,Dict{NamedTuple,Float64}}

# What a role's output is compared with in the verification problem's reference: `dbias_t` is the
# bias gradient transposed.
_reference(role::Symbol, ref) = role === :dbias_t ? ref.dbias : getfield(ref, role)
_as_reference(role::Symbol, x) = role === :dbias_t ? permutedims(x, (2, 1, 3)) : x

"""
    verify_stage(kernel, stage, b, V) -> Bool

The correctness gate: run binary `b` on [`verification_problem`](@ref) `V` and compare every output
of the stage with the plain-loop reference, within [`verification_tolerance`](@ref). A backward stage
is also checked on a large `dO` ([`DO_SCALE`](@ref)).
"""
function verify_stage(k::BuiltinKernel, stage::Symbol, b::KernelDispatch.KernelBinary, V)
    stage === :fwd && return _verify_on(k, stage, b, V)
    return _verify_on(k, stage, b, V) && _verify_on(k, stage, b, _scaled_dO(V))
end

function _verify_on(k::BuiltinKernel, stage::Symbol, b::KernelDispatch.KernelBinary, V)
    T = eltype(V.q)
    P = packed_problem(V.q, V.k, V.v, V.bias, V.mask, V.dO, V.o, V.lse, V.delta)
    outs = run_stage!(k, stage, b, P; V.n, V.nouter, heads=V.bh, V.mheads)
    err = maximum(zip(stage_outputs(k, stage), outs)) do (role, x)
        verification_error(_as_reference(role, Array(x)), _reference(role, V.ref))
    end
    return err <= verification_tolerance(T)
end

"""
    DO_SCALE

The large-`dO` sanity check of the backward stages: the verification problem again with `dO` (and so
`delta` and every gradient) scaled by 256, a power of two, exact in every storage type; its largest
`dO` entry is then ~1e3. The errors are relative, so the scale cancels.

What it does NOT decide: whether a Float16 `dS = P·(dP − δ)` can overflow is a property of the INPUTS
— it needs peaked attention (`P` near 1/2…1) together with a large aligned `|v|·|dO|` (`|dP| ≤ dim ·
max|v| · max|dO|`) — not of the configuration. On this random problem `P` is spread over ~67 keys and
Float16 `dS` stays far from overflow even at `dO` ~ 2e4 (measured on an M1 Pro). Float16 operands are
the FlashAttention-2 convention and carry its caveat: a loss-scaled `dO` large enough to push `dS` past
65504 gives non-finite gradients, as it would there.
"""
const DO_SCALE = 256

function _scaled_dO(V, s=DO_SCALE)
    T = eltype(V.dO)
    r = V.ref
    return merge(V, (; dO=V.dO .* T(s), delta=V.delta .* Float32(s),
        ref=merge(r, (; dq=r.dq .* s, dk=r.dk .* s, dv=r.dv .* s, dbias=r.dbias .* s))))
end

# The GPU time of one launch, per dispatch: `reps` back-to-back dispatches in one command buffer
# (the kernel alone, not the synchronous launch's 3–7 ms of latency), the better of two, the first
# after one warm-up.
_gpu_time(l::KernelDispatch.Launch; reps::Integer=3) =
    minimum(KernelDispatch.time_binary(l; reps, warmup=i == 1 ? 1 : 0) for i in 1:2)

# Verification does not depend on the length: done once per candidate and session.
const _VERIFIED = Dict{Any,Bool}()

"""
    measure(kernel, stage, T, dim, p; V, probe) -> Float64

One candidate: build it, check its pipeline admits its threads, verify it on `V` (once per
session), then the GPU time of one launch on the probe (`probe = (; P, n, nouter, heads)`). `Inf`
for a candidate that fails any step or throws.
"""
function measure(k::BuiltinKernel, stage::Symbol, ::Type{T}, dim::Integer, p::NamedTuple;
    V, probe) where {T}
    try
        b = build_stage(k, stage, T, dim, p)
        b.threadgroup <= KernelDispatch.max_threads(b) || return Inf
        ok = get!(_VERIFIED, (KernelDispatch.kernelname(k), stage, T, Int(dim), p)) do
            v = verify_stage(k, stage, b, V)
            v || @warn "$(KernelDispatch.kernelname(k)) configuration failed verification; excluded" stage p
            v
        end
        ok || return Inf
        l = stage_launch(k, stage, b, probe.P; probe.n, probe.nouter, probe.heads, mheads=probe.heads)
        return _gpu_time(l)
    catch err
        @debug "configuration failed" kernel = KernelDispatch.kernelname(k) stage p exception = err
        return Inf
    end
end

# A probe's outer extent and head count, capped so each `[dim, n, nouter, heads]` buffer stays near
# 32M elements (~128 MB at Float32; the probe holds about ten). The cap only ever shrinks the outer
# extent, then the head count: a probe that large already saturates the GPU.
function _probe_geometry(dim, n, nouter, heads; budget=1 << 25)
    nouter = max(1, min(nouter, budget ÷ (dim * n * heads)))
    heads = max(1, min(heads, budget ÷ (dim * n * nouter)))
    return nouter, heads
end

function _probe(::Type{T}, dim, n, heads) where {T}
    nouter, heads = _probe_geometry(dim, n, n, heads)
    rng = Random.Xoshiro(1)
    r(dims...) = T.(randn(rng, Float32, dims...))
    q, k, v, dO = (r(dim, n, nouter, heads) for _ in 1:4)
    P = packed_problem(q, k, v, r(n, n, heads), fill(false, n, nouter, 1), dO,
        zeros(T, dim, n, nouter, heads), zeros(Float32, n, nouter, heads), zeros(Float32, n, nouter, heads))
    return (; P, n, nouter, heads)
end

# --- calibration: measured once per (kernel, device, dim, T), persisted beside the kernel ------------

# Bumped whenever what a calibration MEANS changes, so an old file is re-measured rather than misread.
const CALIBRATION_VERSION = 5

const _CALIBRATIONS = Dict{Any,Calibration}()

# A parameter value may be a type (`:flash_mma`'s operand type `ops`): stored by name.
const _TYPE_NAMES = Dict("Float32" => Float32, "Float16" => Float16, "BFloat16" => BFloat16)
_to_toml(x::Type) = String(nameof(x))
_to_toml(x) = x
_from_toml(x::String) = _TYPE_NAMES[x]
_from_toml(x) = x

_slug(s::AbstractString) = replace(strip(s), r"[^A-Za-z0-9._-]+" => "_")

"""
    calibration_path(kernel, device, dim, T, n) -> String

Where a calibration is persisted: beside the kernel's registered binaries,
`<KernelDispatch.cache_dir(SDPA())>/<kernelname>/calibration/<device>__d<dim>__<T>__n<n>.toml`. TOML, so
it can be read, diffed and hand-edited.
"""
calibration_path(k::BuiltinKernel, device::AbstractString, dim::Integer, ::Type{T}, n::Integer) where {T} =
    joinpath(KernelDispatch.cache_dir(SDPA()), String(KernelDispatch.kernelname(k)), "calibration",
        "$(_slug(device))__d$(dim)__$(lowercase(string(nameof(T))))__n$(n).toml")

"""
    save_calibration(kernel, path, tref)

Persist a calibration: per stage, its axes and one entry per finite point (`values`, `time`).
Non-finite entries (points that failed to build, dispatch or verify) are dropped, as TOML
cannot represent `Inf`.
"""
function save_calibration(k::BuiltinKernel, path::AbstractString, tref::Calibration)
    mkpath(dirname(path))
    tbl = Dict{String,Any}("version" => CALIBRATION_VERSION, "source_tag" => KernelDispatch.source_tag(k))
    for (stage, pts) in tref
        fin = [(p, t) for (p, t) in pts if isfinite(t)]
        isempty(fin) && continue
        tbl[String(stage)] = Dict{String,Any}(
            "axes" => collect(String.(keys(first(fin)[1]))),
            "entry" => [Dict{String,Any}("values" => Any[_to_toml(x) for x in values(p)], "time" => t)
                        for (p, t) in fin])
    end
    open(io -> TOML.print(io, tbl), path, "w")
    return path
end

"""
    load_calibration(kernel, path) -> Calibration or `nothing`

Read a persisted calibration, or `nothing` if it is absent, unreadable, of another
`CALIBRATION_VERSION` or kernel source, or missing a stage. A stale cache must never be
fatal: the response is to measure again.
"""
function load_calibration(k::BuiltinKernel, path::AbstractString)
    isfile(path) || return nothing
    return try
        raw = TOML.parsefile(path)
        (get(raw, "version", 0) == CALIBRATION_VERSION &&
         get(raw, "source_tag", "") == KernelDispatch.source_tag(k)) || return nothing
        tref = Calibration()
        for stage in tune_stages(k)
            tbl = get(raw, String(stage), nothing)
            tbl === nothing && return nothing
            names = Tuple(Symbol.(tbl["axes"]))
            tref[stage] = Dict{NamedTuple,Float64}(
                NamedTuple{names}(Tuple(_from_toml.(e["values"]))) => Float64(e["time"]) for e in tbl["entry"])
            isempty(tref[stage]) && return nothing
        end
        tref
    catch
        nothing
    end
end

"""
    calibrate!(kernel, T=Float32; dim, n, force=false, probe_heads=4, persist=true)
        -> Dict(stage => Dict(point => seconds))

Measure every stage of a built-in kernel on this device at sequence length `n`: the admissible
points of its [`tune_grid`](@ref), searched ([`search`](@ref)) and each [`measure`](@ref)d — built,
admitted by its pipeline, verified, timed on a triangle probe of `n` queries (`nouter = n`, capped
for memory) and `probe_heads` head slices. Persisted ([`calibration_path`](@ref)), so a later call
is a file read; `force=true` re-measures.

Keyed by (kernel, device, dim, `T`, `n`): `T` is the STORAGE type, part of the key because it
changes which configurations are legal (the staged tiles are held in storage precision), not merely
how fast they are. A stage with no finite point is an error — there is then no kernel to run.
"""
function calibrate!(k::BuiltinKernel, ::Type{T}=Float32; dim::Integer, n::Integer,
    force::Bool=false, probe_heads::Integer=4, persist::Bool=true) where {T}
    device_kernel_available() || throw(ErrorException(
        "no usable device backend is loaded, so `:$(KernelDispatch.kernelname(k))` cannot be calibrated. " *
        "Load the package for your GPU (e.g. `using Metal`) first."))
    _check_eltype(T)
    caps = device_caps(KernelDispatch.device_backend())
    key = (KernelDispatch.kernelname(k), caps.name, Int(dim), T, Int(n))
    (!force && haskey(_CALIBRATIONS, key)) && return _CALIBRATIONS[key]
    path = calibration_path(k, caps.name, dim, T, n)
    if !force
        cached = load_calibration(k, path)
        cached === nothing || return (_CALIBRATIONS[key] = cached)
    end

    V = verification_problem(dim, T)
    probe = _probe(T, Int(dim), Int(n), Int(probe_heads))   # up to ~0.5 GiB, only when measuring
    tref = Calibration()
    for stage in tune_stages(k)
        measured = Dict{NamedTuple,Float64}()
        search(grid_points(k, stage, dim, T, caps), p -> (measured[p] = measure(k, stage, T, dim, p; V, probe)))
        any(isfinite, values(measured)) || error(
            "no dispatchable, verified `:$(KernelDispatch.kernelname(k))` configuration for stage " *
            "`:$stage`, dim=$dim, T=$T on $(caps.name)")
        tref[stage] = measured
    end
    _CALIBRATIONS[key] = tref
    persist && save_calibration(k, path, tref)
    return tref
end

"""
    choose_config(t) -> NamedTuple

One stage's configuration from its calibration at a length (`calibrate!(…)[stage]`): the fastest
measured point, or, given a predicate `usable(point)`, the fastest one it accepts.
"""
function choose_config(t::AbstractDict{<:NamedTuple,Float64}, usable=Returns(true))
    fin = sort!([(p, x) for (p, x) in t if isfinite(x)]; by=last)
    for (p, _) in fin
        usable(p) && return p
    end
    error("no usable configuration in this calibration")
end

# The binaries `tune!` registers per stage: every bias / mask presence, each compiled on its own.
const _SPECIALISATIONS = ((true, true), (true, false), (false, true), (false, false))

# Whether `p` is dispatchable in EVERY specialisation of `stage`. `measure` checked the general one (bias and mask
# present); an absent-operand build uses other registers, so its pipeline can admit fewer threads (BFloat16 `:dkv`
# without a bias: 448 against the 512 threads that were measured), and launching over the limit fails.
_admits_all(k::BuiltinKernel, stage::Symbol, ::Type{T}, dim::Integer, p::NamedTuple) where {T} =
    all(_SPECIALISATIONS) do (hb, hm)
        b = build_stage(k, stage, T, dim, p; hasbias=hb, hasmask=hm)
        b.threadgroup <= KernelDispatch.max_threads(b)
    end

# The point `tune!` registers for `stage`: the fastest measured one that every specialisation admits.
_choose_stage(k::BuiltinKernel, stage::Symbol, ::Type{T}, dim::Integer, t) where {T} =
    choose_config(t, p -> _admits_all(k, stage, T, dim, p))

"""`tuned_config(kernel, T, dim, n) -> Dict(stage => point)`: what [`tune!`](@ref) picks at length `n`."""
tuned_config(k::BuiltinKernel, ::Type{T}, dim::Integer, n::Integer) where {T} =
    Dict(s => _choose_stage(k, s, T, dim, t) for (s, t) in calibrate!(k, T; dim, n))

"""
    tune_lengths() -> Vector{Int}

The sequence lengths the built-in kernels are tuned for when no length is given — by
[`tune_flash_kernel!`](@ref) / [`tune_flash_mma!`](@ref) and by the first device call that finds no
variant. Default `[256, 512]`, which suits most binder-design work; set it per project with
[`set_tune_lengths!`](@ref) (e.g. `[256, 512, 768, 1024]` for larger proteins).
"""
tune_lengths() = Int.(Preferences.load_preference(@__MODULE__, "tune_lengths", [256, 512]))

"""
    set_tune_lengths!(lengths)

Persist the default [`tune_lengths`](@ref) in the active project's `LocalPreferences.toml`. Takes
effect immediately; already tuned lengths stay registered.
"""
function set_tune_lengths!(lengths)
    ls = sort!(unique!(Int.(collect(lengths))))
    (isempty(ls) || any(<=(0), ls)) && throw(ArgumentError("tune lengths must be positive; got $lengths"))
    Preferences.set_preferences!(@__MODULE__, "tune_lengths" => ls; force=true)
    return ls
end

"""
    tune!(kernel, lengths=tune_lengths(); dim, T=Float32, force=false) -> variants

Tune, compile and register a built-in kernel for each sequence length in `lengths` (an integer or a
collection). For each length `n`:

 1. calibrate the device at `n` for `(dim, T)`, or reuse the persisted calibration ([`calibrate!`](@ref));
 2. pick, PER STAGE, the fastest configuration at `n` ([`choose_config`](@ref));
 3. compile every stage at its configuration, for every bias / mask presence (the absent-operand
    specialisations are binaries of their own, so a traced call without a bias or mask never has to
    compile), and register them with `KernelDispatch.add_variant!`, keyed `(dtype, dim, n, hasbias, hasmask)`.

A call of any length runs the variant tuned for the closest one (`KernelDispatch.nearest`). A length with
a registered variant is skipped unless `force` (which also re-measures). The attention geometry
(outer extent, heads) is not a key: on an M1 Pro, over triangle, MSA-row and plain-sequence shapes,
the per-device choice costs at most ~10% against tuning each geometry (docs/done/FEATURE_flash_mma.md).

Seconds to minutes per length on a cold calibration; run it deliberately, ahead of latency-sensitive
work.
"""
function tune!(k::BuiltinKernel, lengths=tune_lengths(); dim::Integer, T::Type=Float32, force::Bool=false)
    vs = KernelDispatch.variants(SDPA(), k)
    for n in (lengths isa Integer ? (lengths,) : lengths)
        keys_ = [_attention_variant_key(k, T, dim, n; hasbias=hb, hasmask=hm) for hb in (true, false)
                 for hm in (true, false)]
        (!force && all(in(keys(vs)), keys_)) && continue
        cfgs = Dict(s => _choose_stage(k, s, T, dim, t) for (s, t) in calibrate!(k, T; dim, n, force))
        for (hb, hm) in _SPECIALISATIONS
            bins = [s => build_stage(k, s, T, dim, cfgs[s]; hasbias=hb, hasmask=hm) for s in tune_stages(k)]
            KernelDispatch.add_variant!(SDPA(), k, _attention_variant_key(k, T, dim, n; hasbias=hb, hasmask=hm);
                bins...)
        end
        @debug "tuned" kernel = KernelDispatch.kernelname(k) n dim T configs = cfgs
    end
    return vs
end

"""
    clear_calibration!(; disk=false)

Drop cached calibrations. In-memory only by default; `disk=true` also removes the persisted files of
both built-in kernels, which is what you want after changing a kernel, since a stale calibration would
then measure something that no longer exists (a changed source tag also invalidates it).
"""
function clear_calibration!(; disk::Bool=false)
    empty!(_CALIBRATIONS)
    empty!(_VERIFIED)
    disk && for k in (FlashScalar(), FlashMMA())
        rm(joinpath(KernelDispatch.cache_dir(SDPA()), String(KernelDispatch.kernelname(k)), "calibration");
            recursive=true, force=true)
    end
    return nothing
end
