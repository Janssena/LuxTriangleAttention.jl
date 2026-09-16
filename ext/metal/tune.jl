# Metal's half of the tuning story: the device's real limits, and the timing callback.
#
# The POLICY — which configurations are candidates, how `n` enters the cost, how a choice is made —
# lives once in `src/kernels/flash/tune.jl`. Metal contributes only the two things that policy
# cannot know: what this device reports, and how long a launch actually takes here.

"""
    occupancy_limit(T, dim, tg, kt) -> Int

Threads per group this **compiled pipeline** admits — not the device cap, and the gap is not
academic. The kernel holds `2*DIM` floats per thread, which at dim=32 drops the limit to 384 against
a device maximum of 1024. Over-dispatching does not raise: it returns garbage *and* an impossibly
fast time (TG=512 "measured" 0.9 ms against a true 93 ms), which reads exactly like a tuner finding
a huge win. This is the filter `candidate_configs` requires.
"""
occupancy_limit(::Type{T}, dim::Int, tg::Int, kt::Int) where {T} =
    Int(pipeline(build(T, dim, tg, kt)...).maxTotalThreadsPerThreadgroup)

dispatchable(::Type{T}, dim::Int, tg::Int, kt::Int) where {T} =
    tg <= occupancy_limit(T, dim, tg, kt)

device_caps() = LTA.DeviceCaps(32, 1024,
    Int(Metal.device().maxThreadgroupMemoryLength), String(Metal.device().name))

function _probe(::Type{T}, n, heads, dim) where {T}
    mk4() = MtlArray(rand(T, dim, n, n, heads))
    return (; q=mk4(), k=mk4(), v=mk4(),
        bias=MtlArray(rand(T, n, n, heads)),
        mask=MtlArray(falses(n, n)),
        o=MtlArray(zeros(T, dim, n, n, heads)),
        lse=MtlArray(zeros(Float32, n, n, heads)))   # residual stays Float32
end

function LTA.calibrate!(::Type{T}=Float32; dim::Int, force::Bool=false,
    probe_n::Int=512, heads::Int=2, reps::Int=5) where {T}
    LTA.device_kernel_available() ||
        error("no functional Metal device; cannot calibrate")
    _check_eltype(T)
    # The probe is ~262 MiB of device buffers at the default size, and it is needed ONLY to time
    # candidate configurations during a real calibration. `LTA.calibrate` returns early from its
    # in-memory or on-disk cache on the common path, so allocating eagerly spent that memory — plus
    # the time to fill three 64 MB random arrays — on every call, including the overwhelmingly
    # common cache hit. Built on first use instead: on a cache hit `time_config` is never invoked
    # and nothing is allocated.
    probe = Ref{Any}(nothing)
    _probe!() = (probe[] === nothing && (probe[] = _probe(T, probe_n, heads, dim)); probe[])
    function time_config(tg, kt, n)
        p = _probe!()
        go() = run_fwd!(p.o, p.lse, p.q, p.k, p.v, p.bias, p.mask;
            tg, kt, n, heads, dim)
        go()
        return minimum(@elapsed(go()) for _ in 1:reps)
    end
    return LTA.calibrate(device_caps(), dim, T; time_config,
        dispatchable=(tg, kt) -> dispatchable(T, dim, tg, kt), probe_n, force)
end
