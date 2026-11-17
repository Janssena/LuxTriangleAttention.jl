# Backward-kernel sweep: runtime AND peak memory for the reverse rule, over N, at Boltz2's real
# triangle-attention shape. The forward analogue is `test/bench/kernels/forward.bench.jl` — same subprocess-per-
# cell design, same shape, same CSV mechanics; see its header for the rationale behind both. This
# file only restates what differs for the backward pass.
#
#     julia --project=test test/bench/kernels/backward.bench.jl              # default sweep, N = 64:64:960
#     LTA_BENCH_NS=64:64:256 julia --project=test test/bench/kernels/backward.bench.jl
#
# ## What's compared, and what's deliberately left out
#
# Three paths, not five: `SDPA()`'s CPU/no-kernel fallback is plain `_sdpa_default` (see
# `src/kernels/sdpa_op.jl`), which has no reverse rule of its own — Enzyme differentiates it exactly
# like any other ordinary call — so there is no separate "julia_native" backward to measure, only the
# one Enzyme/XLA path below. There is likewise no `xla_gpu_ka` cell here (the traced custom_call
# reverse rule, forward's `xla_gpu_ka` counterpart): the two direct-Metal cells below already isolate
# kernel-vs-kernel, which is the comparison this file exists for.
#
#   * `xla_cpu_bwd`      — `Enzyme.gradient` of `sum(_sdpa_default(...))`, through `Reactant.@compile`
#                          on the CPU client. This is what a model's training step actually runs on
#                          CPU: ordinary HLO, Enzyme walking straight through the softmax, no custom
#                          VJP.
#   * `metal_ka_bwd`     — registry kernel `:default`: our own compiled backward
#                          (`flash_bwd_dq!`/`dk!`/`dv!`, three kernels for the register-pressure
#                          reasons documented in `src/kernels/flash/backward.jl`), via eager
#                          `sdpa_bwd` on `MtlArray`s, no XLA in the path.
#   * `metal_trifast_bwd` — registry kernel `:trifast`: trifast's Triton-generated backward, via the
#                          SAME `sdpa_bwd` call. Its stages are `_bwd_q` (also writes
#                          `delta`), `_bwd_kv`, `_bwd_b`; ours folds `dbias` into `flash_bwd_dq!` via
#                          atomics instead of a dedicated kernel — both cells compute the SAME four
#                          outputs (`dq`, `dk`, `dv`, `dbias`), just partitioned across a different
#                          number of kernels, which is exactly the comparison worth making.
#
# The two Metal cells differ ONLY in which kernel object `kernel_named` picks: same launcher, same
# `[hd, H, N, N, B]` operands read in place, same explicit all-keep mask and bias, same per-call
# launcher work, and ONE configuration for every `N` each (see `registry_setup.jl`). The argument
# order, grids and tiles of trifast's kernels are declared once, as roles, in `register_trifast!`.
#
# Every cell first runs its own forward (through the same registry entry) to produce the `o`/`lse`
# residual its backward needs — that cost is NOT part of the timed closure (see `worker_body`), only
# the backward itself is.

using Printf, Dates
# Private kernel registry + matched tuning for the Metal kernels; must precede `using
# LuxTriangleAttention` (see its header).
include(joinpath(@__DIR__, "registry_setup.jl"))
using LuxTriangleAttention, Reactant, Enzyme
include(joinpath(@__DIR__, "..", "..", "support", "sdpa_bwd.jl"))   # sdpa_bwd

const HEAD_DIM = 32   # Boltz2 pairwise_head_width
const N_HEADS = 4     # Boltz2 pairwise_num_heads
const BATCH = 1       # one protein at a time

const METHODS = [
    (key="xla_cpu_bwd", label="LuxLib SDPA-XLA (bwd)", metal=false),
    (key="metal_ka_bwd", label="Metal registry (KA bwd)", metal=true),
    (key="metal_trifast_bwd", label="Metal registry (trifast bwd)", metal=true),
]

const BENCH_NS = collect(eval(Meta.parse(get(ENV, "LTA_BENCH_NS", "64:64:960"))))
const TIMEOUT_S = parse(Float64, get(ENV, "LTA_BENCH_TIMEOUT", "900"))

qkv_bytes(N) = 4 * HEAD_DIM * N_HEADS * N^2 * BATCH

human(b) = b < 1024 ? @sprintf("%dB", b) :
           b < 1024^2 ? @sprintf("%.1fK", b / 1024) :
           b < 1024^3 ? @sprintf("%.1fM", b / 1024^2) : @sprintf("%.2fG", b / 1024^3)
human_t(s) = s < 1e-3 ? @sprintf("%.1fus", s * 1e6) :
             s < 1 ? @sprintf("%.1fms", s * 1e3) : @sprintf("%.2fs", s)

inputs(N) = (randn(Float32, HEAD_DIM, N_HEADS, N, N, BATCH),
    randn(Float32, HEAD_DIM, N_HEADS, N, N, BATCH),
    randn(Float32, HEAD_DIM, N_HEADS, N, N, BATCH),
    randn(Float32, N, N, N_HEADS, 1, BATCH))


# =================================================================================================
# Worker: one method, one N, one process
# =================================================================================================

# Each `build_*` returns a zero-argument closure that runs the backward once. Anything before the
# closure — forward residuals, calibration, pipeline compilation — is setup cost, not measured.

function build_xla_cpu_bwd(N)
    Reactant.set_default_backend("cpu")
    q, k, v, bias = Reactant.to_rarray.(inputs(N))
    lref(a, b, c, d) = sum(LuxTriangleAttention._sdpa_default(a, b, c, nothing, d))
    gradf(a, b, c, d) = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(lref), a, b, c, d)
    c_ = Reactant.@compile sync = true gradf(q, k, v, bias)
    return () -> c_(q, k, v, bias)
end

# --- registry paths -----------------------------------------------------------------------------
# Both Metal kernels are registry entries run through the same eager `sdpa_bwd` call; the
# only difference between the two cells is the kernel object (`kernel_named`). See `test/bench/kernels/forward.bench.jl`.

"This package's own `[hd, H, N, N, B]` operands on the device, plus `dO` and an all-keep mask."
function metal_inputs(N)
    Metal = Main.Metal
    mk() = Metal.MtlArray(randn(Float32, HEAD_DIM, N_HEADS, N, N, BATCH))
    return (; q=mk(), k=mk(), v=mk(), dO=mk(),
        bias=Metal.MtlArray(randn(Float32, N, N, N_HEADS, BATCH)),
        mask=Metal.MtlArray(trues(N, N, BATCH)))       # KEEP mask: dense, as in the forward sweep
end

"""
    registry_backward(name, N)

A closure running the selected registry kernel's backward once. Its forward runs here, once and
unmeasured, through the same entry, to produce the `o`/`lse` residual.
"""
function registry_backward(name::Symbol, N)
    LTA = LuxTriangleAttention
    kern, p = kernel_named(name), metal_inputs(N)
    o, lse = KernelDispatch.forward(LTA.SDPA(), kern, p.q, p.k, p.v, p.mask, p.bias)
    return () -> sdpa_bwd(p.q, p.k, p.v, p.mask, p.bias, o, lse, p.dO; kernel=kern)
end

build_metal_ka_bwd(N) = (tune_default_fixed!(; dim=HEAD_DIM); registry_backward(:default, N))
build_metal_trifast_bwd(N) = (register_trifast!(; dim=HEAD_DIM); registry_backward(:trifast, N))

const BUILDERS = Dict(
    "xla_cpu_bwd" => build_xla_cpu_bwd,
    "metal_ka_bwd" => build_metal_ka_bwd, "metal_trifast_bwd" => build_metal_trifast_bwd)

"""
    worker(method, N)

Run one cell and print a single `RESULT` line. Everything this process does — loading, compiling,
allocating, running — is inside the peak-RSS reading, by design (see `test/bench/kernels/forward.bench.jl`'s
header).
"""
function worker(method::AbstractString, N::Int)
    only(m.metal for m in METHODS if m.key == method) && @eval using Metal
    # `invokelatest`: the Metal extension's methods (`KernelDispatch.to_device`, `device_caps`, …) are defined by the
    # `@eval using` above and are invisible to this function's already-fixed world age.
    return Base.invokelatest(worker_body, method, N)
end

"""
    device_allocated() -> Int

Metal's own allocation counter, or `0` where there is no device in the process. See
`test/bench/kernels/forward.bench.jl`'s docstring of the same name for why this, not RSS, is the real number for
a unified-memory GPU.
"""
device_allocated() = isdefined(Main, :Metal) ? Int(Main.Metal.device().currentAllocatedSize) : 0

function worker_body(method, N)
    GC.gc(true)
    baseline = Sys.maxrss()   # packages are a fixed harness cost, not the method's
    dev_base = device_allocated()

    run_once = BUILDERS[method](N)
    run_once()   # warm up / compile

    dev_peak = device_allocated()

    times = Float64[]
    t0 = time_ns()
    while length(times) < 3 || ((time_ns() - t0) * 1e-9 < 2.0 && length(times) < 50)
        t = time_ns()
        run_once()
        push!(times, (time_ns() - t) * 1e-9)
    end

    peak = Sys.maxrss()
    @printf("RESULT\t%s\t%d\t%.9f\t%d\t%d\t%d\n", method, N,
        sort(times)[(length(times)+1)÷2], peak, max(peak - baseline, 0),
        max(dev_peak - dev_base, 0))
    return nothing
end


# =================================================================================================
# Driver
# =================================================================================================

const RESULTS_CSV = joinpath(@__DIR__, "results.kernels.bwd.csv")
const CSV_HEADER = "timestamp,method,label,N,head_dim,heads,batch,seconds," *
                   "peak_rss_bytes,delta_rss_bytes,device_bytes,status"

"""
    csv_init!()
    csv_row!(m, N, r)

Append one row per cell, as it completes — see `test/bench/kernels/forward.bench.jl`'s docstring of the same name
for why (crash resilience, and the CSV as the live progress view while output is piped).
"""
csv_init!() = open(io -> println(io, CSV_HEADER), RESULTS_CSV, "w")

function csv_row!(m, N, r)
    open(RESULTS_CSV, "a") do io
        @printf(io, "%s,%s,%s,%d,%d,%d,%d,%s,%d,%d,%d,%s\n",
            Dates.format(Dates.now(), "HH:MM:SS"), m.key, m.label, N,
            HEAD_DIM, N_HEADS, BATCH, r.ok ? @sprintf("%.9f", r.time) : "",
            r.peak, r.delta, r.dev, r.note)
        flush(io)
    end
    return nothing
end

"""
    run_cell(method, N) -> NamedTuple

Spawn one worker and parse its `RESULT` line. See `test/bench/kernels/forward.bench.jl`'s docstring of the same
name for the crash/timeout handling, identical here.
"""
function run_cell(method::AbstractString, N::Int)
    cmd = `$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..", "..")) $(@__FILE__) --worker $method $N`
    out = IOBuffer()
    p = run(pipeline(cmd; stdout=out, stderr=devnull); wait=false)
    timer = Timer(_ -> (process_running(p) && kill(p, Base.SIGKILL)), TIMEOUT_S)
    wait(p)
    close(timer)
    ok = success(p)
    for l in split(String(take!(out)), '\n')
        startswith(l, "RESULT\t") || continue
        f = split(l, '\t')
        return (; ok=true, time=parse(Float64, f[4]),
            peak=parse(Int, f[5]), delta=parse(Int, f[6]), dev=parse(Int, f[7]), note="ok")
    end
    return (; ok=false, time=NaN, peak=0, delta=0, dev=0,
        note=ok ? "no-result" : "OOM/timeout")
end

function main()
    @printf("Triangle-attention BACKWARD sweep — Boltz2 shape: head_dim=%d, heads=%d, B=%d\n",
        HEAD_DIM, N_HEADS, BATCH)
    @printf("q/k/v/dO [%d, %d, N, N, %d]; bias [N, N, %d, 1, %d]; DENSE (no mask: B=1, no padding)\n",
        HEAD_DIM, N_HEADS, BATCH, N_HEADS, BATCH)
    println("Columns: median time | peak RSS over the process baseline | Metal device allocation.")
    println("Each cell's own forward runs once, unmeasured, to produce the o/lse residual; only")
    println("the backward (dQ, dK, dV, dBias) is timed. A method that fails is skipped at every")
    println("larger N. Live results appended per cell to $(RESULTS_CSV)")
    println()

    csv_init!()
    dead = Dict{String,Int}()       # method => N at which it first failed
    for N in BENCH_NS
        @printf("N = %-5d (q/k/v %s each)\n", N, human(qkv_bytes(N)))
        flush(stdout)
        for m in METHODS
            if haskey(dead, m.key)
                skipped = (; ok=false, time=NaN, peak=0, delta=0, dev=0,
                    note="skipped(died@$(dead[m.key]))")
                csv_row!(m, N, skipped)
                @printf("    %-24s %10s   %10s   %10s\n", m.label, "-",
                    "skip (died @ $(dead[m.key]))", "-")
                flush(stdout)
                continue
            end
            r = run_cell(m.key, N)
            csv_row!(m, N, r)                       # written before anything else can fail
            r.ok || (dead[m.key] = N)
            @printf("    %-24s %10s   %10s   %10s\n", m.label,
                r.ok ? human_t(r.time) : "-", r.ok ? human(r.delta) : r.note,
                (r.ok && r.dev > 0) ? human(r.dev) : "-")
            flush(stdout)
        end
        println()
    end
    println("wrote $RESULTS_CSV")
    return nothing
end


if !isempty(ARGS) && ARGS[1] == "--worker"
    worker(ARGS[2], parse(Int, ARGS[3]))
else
    main()
end
