# Forward-kernel sweep: runtime AND peak memory for every SDPA path, over N, at Boltz2's real
# triangle-attention shape.
#
#     julia --project=test test/bench/kernels/forward.bench.jl              # default sweep, N = 64:64:960
#     LTA_BENCH_NS=64:64:256 julia --project=test test/bench/kernels/forward.bench.jl
#
# ## Shape
#
# Boltz2's pairformer builds triangle attention from `chn_token_z=128`, `pairwise_head_width=32`,
# `pairwise_num_heads=4` (`lib/Boltz2/src/pairformer/pairformer_layer.jl`), so `q`/`k`/`v` are
# `[32, 4, N, N, 1]` and the pair bias is `[N, N, 4, 1, 1]`. `B=1` is one protein at a time, which is
# the real inference case; `N` is the token count, swept in steps of 64.
#
# **No mask, i.e. DENSE.** With `B=1` there is no padding, so the pair mask is all-true and every
# key is computed. This matters for comparing against older numbers: the earlier scratchpad harness
# benchmarked with `rand(Bool, n, n)` — a ~50% mask — and the kernels skip masked keys entirely, so
# those figures are ~1.55x faster than the dense case and do NOT transfer. Every method here gets
# the same dense problem.
#
# ## Why every point runs in a fresh subprocess
#
# Memory is the *point* of this benchmark, and it cannot be measured honestly in one process:
#
#   * `Sys.maxrss()` is a high-water mark that never decreases, so after the first method every
#     later reading is contaminated by the largest one before it.
#   * MLX (behind jax-mps) reserves ~1.5x its working set and does not return freed buffers between
#     unrelated compiles, so one GPU point poisons every CPU point after it.
#   * The unfused path materialises an `[Nk, Nq, H, Nj, B]` score tensor — `16·N³` bytes, 2 GB at
#     N=512 and 13.7 GB at N=960 — and is *expected* to die partway up the sweep. In-process that
#     would take the whole sweep with it; as a subprocess it is one `OOM` cell.
#
# Once a method fails at some N it is **skipped at every larger N** — it will not recover, and the
# retries are slow and can thrash the machine.
#
# Two memory numbers are reported per cell, because neither answers the other's question:
#
#   * **RSS** — the worker's peak over its own post-load baseline, INCLUDING compilation. It is what
#     has to fit in RAM, but it carries a large fixed floor (~800 MB for the GPU paths) and it
#     UNDERSTATES Metal's unified memory, so it is not comparable to an XLA/MLX arena figure.
#     Compare its marginal growth across N, never its absolute value.
#   * **Device** — `MTLDevice.currentAllocatedSize`, Metal's own counter, for the GPU methods. This
#     is the real working set. RSS did not merely blur these: at N=832 it read 881 MB against
#     1.42 GB of live tensors, and it ranked the three Metal kernels in the wrong order.
#
# ## Two things that would quietly invalidate the numbers
#
# 1. `Reactant` is loaded unconditionally rather than per method — `@compile`/`@code_hlo` are macros,
#    so the binding has to exist when this file is lowered, not when a worker runs. It is a fixed
#    cost the baseline subtracts. `Metal` is loaded only for the GPU methods, since a Metal context
#    costs resident memory the CPU paths should not be charged for.
# 2. `_ka_flash_emit` declines for many ordinary reasons (dtype, rank, no calibration, no binary) and
#    the host fallback is numerically identical — so a decline looks exactly like a slow GPU result.
#    The `xla_gpu_ka` cell asserts `mps.metal_kernel_lib` is actually in the emitted IR and fails
#    loudly otherwise. Set `LTA_MPS_DEBUG=1` to see why it declined.

using Printf, Dates
# Private kernel registry + matched tuning for the Metal kernels; must precede `using
# LuxTriangleAttention` (see its header).
include(joinpath(@__DIR__, "registry_setup.jl"))
using LuxTriangleAttention, Reactant

include(joinpath(@__DIR__, "..", "..", "mps_client.jl"))

const HEAD_DIM = 32   # Boltz2 pairwise_head_width
const N_HEADS = 4     # Boltz2 pairwise_num_heads
const BATCH = 1       # one protein at a time


"""
Every path under test, in report order.

  * `julia_native`  — `_sdpa_default` (= `Lux.scaled_dot_product_attention`) on plain `Array`s.
  * `xla_cpu`       — the same function through `Reactant.@compile` on the CPU client. There used to
                      be an `xla_cpu_flash` cell here too: a hand-tiled, online-softmax forward
                      (`flash_default_fwd`, since removed) lowered to XLA. It existed to isolate what
                      a fused *algorithm* buys before any hand-written kernel, on the theory that
                      tiling would help XLA the same way it helps a real device kernel. Measured
                      instead: XLA's own compiler already fuses the one-shot formulation
                      `_sdpa_default` uses into a single streaming pass with no larger a memory
                      footprint, while `Reactant.@trace` traces a Julia `for` loop by UNROLLING it —
                      multiplying the compiled program's size and memory traffic by the tile count
                      instead of shrinking either. At N=896 the tiled cell needed ~4x the device
                      memory of `xla_cpu` and ran ~30-40x slower, worsening with `N` until it OOM'd
                      outright by N=960 — this is also why `SDPA()`'s CPU fallback is plain
                      `_sdpa_default` rather than a hand-tiled kernel of its own; see its docstring
                      in `src/kernels/sdpa_op.jl`.
  * `xla_gpu_ka`    — `SDPA()` on a Metal client: our KA kernel reached through a
                      `stablehlo.custom_call`, i.e. how a real model would hit it. No mask, so it
                      runs the kernel's mask-free specialisation.
  * `metal_ka`      — registry kernel `:default` (our KA kernel), via eager `SDPA()` on
                      `MtlArray`s, no XLA in the path.
  * `metal_trifast` — registry kernel `:trifast` (trifast's Triton-generated metallibs), via the SAME
                      eager `SDPA()` call.

`metal_ka` and `metal_trifast` differ ONLY in which kernel object `kernel_named` picks: same
launcher, same `[hd, H, N, N, B]` operands read in place, same explicit all-keep mask and bias, same
per-call launcher work (mask negation, output allocation, and for `:default` its transposed bias).
Both are held to ONE configuration for every `N` — trifast's fixed 32×32 tiles, and our kernel tuned
once at `KA_TUNE_N` — see `registry_setup.jl`. `xla_gpu_ka` uses that same single tuning.
"""
const METHODS = [
    (key="julia_native", label="Julia (LuxLib SDPA)", metal=false),
    (key="xla_cpu", label="XLA-CPU (LuxLib SDPA)", metal=false),
    (key="xla_gpu_ka", label="XLA-GPU (KA flash)", metal=true),
    (key="metal_ka", label="Metal registry (KA flash)", metal=true),
    (key="metal_trifast", label="Metal registry (trifast)", metal=true),
]

const BENCH_NS = collect(eval(Meta.parse(get(ENV, "LTA_BENCH_NS", "64:64:960"))))
const TIMEOUT_S = parse(Float64, get(ENV, "LTA_BENCH_TIMEOUT", "900"))

score_bytes(N) = 4 * N^3 * N_HEADS * BATCH       # [Nk, Nq, H, Nj, B] Float32
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

# Each `build_*` returns a zero-argument closure that runs the path once.

function build_julia_native(N)
    q, k, v, bias = inputs(N)
    return () -> LuxTriangleAttention._sdpa_default(q, k, v, nothing, bias)
end

function build_xla_cpu(N)
    Reactant.set_default_backend("cpu")
    q, k, v, bias = Reactant.to_rarray.(inputs(N))
    f(a, b, c, d) = LuxTriangleAttention._sdpa_default(a, b, c, nothing, d)
    c = Reactant.@compile sync = true f(q, k, v, bias)
    return () -> c(q, k, v, bias)
end

function build_xla_gpu_ka(N)
    client = mps_client()
    client === nothing && error("jax-mps Metal PJRT plugin not available")
    Reactant.set_default_backend(client)
    tune_default_fixed!(; dim=HEAD_DIM)      # one tuning for every N, as the direct cells
    q, k, v, bias = Reactant.to_rarray.(inputs(N))
    f(a, b, c, d) = first(LuxTriangleAttention.SDPA()(a, b, c, nothing, d))

    ir = repr(Reactant.@code_hlo optimize = false donated_args = :none f(q, k, v, bias))
    occursin("mps.metal_kernel_lib", ir) ||
        error("device kernel declined; this would have benchmarked the host fallback")
    c = Reactant.@compile sync = true f(q, k, v, bias)
    return () -> c(q, k, v, bias)
end

# --- registry paths -----------------------------------------------------------------------------
# Both Metal kernels are registry entries run through the same eager `KernelDispatch.forward` call; the only
# difference between the two cells is the kernel object (`kernel_named`). Each launch waits for completion, so the
# closure's wall time is the GPU work plus the launcher's own per-call work, identically for both.

"""
    metal_inputs(N) -> NamedTuple

This package's own `[hd, H, N, N, B]` operands on the device — what every layer hands
`SDPA()` — plus an explicit all-keep mask, so both kernels read a mask (see `METHODS`).
"""
function metal_inputs(N)
    Metal = Main.Metal
    mk() = Metal.MtlArray(randn(Float32, HEAD_DIM, N_HEADS, N, N, BATCH))
    return (; q=mk(), k=mk(), v=mk(),
        bias=Metal.MtlArray(randn(Float32, N, N, N_HEADS, BATCH)),
        mask=Metal.MtlArray(trues(N, N, BATCH)))       # KEEP mask: dense, as above
end

"A closure running the selected registry kernel's forward once on `metal_inputs(N)`."
function registry_forward(name::Symbol, N)
    kern, p = kernel_named(name), metal_inputs(N)
    return () -> KernelDispatch.forward(LuxTriangleAttention.SDPA(), kern, p.q, p.k, p.v, p.mask, p.bias)
end

build_metal_ka(N) = (tune_default_fixed!(; dim=HEAD_DIM); registry_forward(:default, N))
build_metal_trifast(N) = (register_trifast!(; dim=HEAD_DIM); registry_forward(:trifast, N))

const BUILDERS = Dict(
    "julia_native" => build_julia_native, "xla_cpu" => build_xla_cpu,
    "xla_gpu_ka" => build_xla_gpu_ka,
    "metal_ka" => build_metal_ka, "metal_trifast" => build_metal_trifast)

"""
    worker(method, N)

Run one cell and print a single `RESULT` line. Everything this process does — loading, compiling,
allocating, running — is inside the peak-RSS reading, by design (see the file header).
"""
function worker(method::AbstractString, N::Int)
    only(m.metal for m in METHODS if m.key == method) && @eval using Metal
    # `invokelatest`: the Metal extension's methods (`KernelDispatch.to_device`, `device_caps`, …) are defined by the
    # `@eval using` above and are invisible to this function's already-fixed world age.
    return Base.invokelatest(worker_body, method, N)
end

"""
    device_allocated() -> Int

Metal's own allocation counter, or `0` where there is no device in the process.

RSS is the wrong instrument for a unified-memory GPU and not merely an imprecise one: at N=832 it
read 881 MB against 1.42 GB of live `q`/`k`/`v`/`o`, and it ranked the three Metal kernels in the
wrong order. This counter is what the numbers below actually mean.
"""
device_allocated() = isdefined(Main, :Metal) ? Int(Main.Metal.device().currentAllocatedSize) : 0

function worker_body(method, N)
    GC.gc(true)
    baseline = Sys.maxrss()   # packages are a fixed harness cost, not the method's
    dev_base = device_allocated()

    run_once = BUILDERS[method](N)
    run_once()   # warm up / compile

    # Device memory is read after exactly ONE run, unlike RSS. A compiled XLA call returns a FRESH
    # output buffer each time and MLX retains freed buffers up to `JAX_MPS_CACHE_LIMIT_BYTES`, so
    # the counter climbs by one output per rep and would report the harness's rep count rather than
    # the method's working set — +134 MB per rep at N=512, unreclaimed by `GC.gc`.
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

const RESULTS_CSV = joinpath(@__DIR__, "results.kernels.fwd.csv")
const CSV_HEADER = "timestamp,method,label,N,head_dim,heads,batch,seconds," *
                   "peak_rss_bytes,delta_rss_bytes,device_bytes,status"

"""
    csv_init!()
    csv_row!(m, N, r)

Append one row per cell, **as it completes**, rather than accumulating and writing at the end.

Two reasons this matters here and not in a fast benchmark: the sweep runs for a long time and a
method is *expected* to be OOM-killed partway up, so a crash — of a worker or of the driver itself —
must not discard everything measured before it. And this file is the only way to see how far along a
running sweep is: the console output is buffered when the process is piped, so the CSV is the live
progress view. `timestamp` is there to show which cell is slow.
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

Spawn one worker and parse its `RESULT` line. A worker that dies — an `OutOfMemoryError`, or the OS
killing it — is reported as a failed cell rather than taking the sweep down with it. Hard-killed
after `TIMEOUT_S`: a cell that thrashes rather than failing cleanly can hold the machine hostage far
longer than the result is worth.
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
    @printf("Triangle-attention forward sweep — Boltz2 shape: head_dim=%d, heads=%d, B=%d\n",
        HEAD_DIM, N_HEADS, BATCH)
    @printf("q/k/v [%d, %d, N, N, %d]; bias [N, N, %d, 1, %d]; DENSE (no mask: B=1, no padding)\n",
        HEAD_DIM, N_HEADS, BATCH, N_HEADS, BATCH)
    println("Columns: median time | peak RSS over the process baseline | Metal device allocation.")
    println("RSS includes compilation and a large fixed floor; the device column is the real")
    println("working set and is measured after ONE run (see `device_allocated`). GPU methods only.")
    println("A method that fails is skipped at every larger N.")
    println("Live results appended per cell to $(RESULTS_CSV)")
    println()

    csv_init!()
    dead = Dict{String,Int}()       # method => N at which it first failed
    for N in BENCH_NS
        @printf("N = %-5d (q/k/v %s each; unfused score tensor %s)\n",
            N, human(qkv_bytes(N)), human(score_bytes(N)))
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
