# The worker side of the layer benchmarks, shared by `triangle_attention.jl` and `triangle_multiplication.jl` here. A worker is
# one process running ONE method (`ARGS[1]`) over every (variant, case, N) the driver, `test/bench_utils.jl`, gives it. See
# there for the protocol and the environment.
#
# Per cell: build the inputs, prepare the method (`prepare_*` return `(run, sync, readout)`: one forward pass, wait for the
# device, the output on the host in Julia's layout), check the output against the reference's, then time it.

using Printf, Random, Statistics, LinearAlgebra, BenchmarkTools

const METHOD = ARGS[1]
const FRAMEWORK, DEVICE, KERNEL = let p = split(METHOD, "_")
    (p[1], p[2], length(p) > 2 && p[3] == "kernel")
end
const NEEDS_GPU = DEVICE == "gpu"
const GPU = Symbol(lowercase(get(ENV, "LTA_BENCH_GPU", Sys.isapple() ? "metal" : "cuda")))
const TIMED = get(ENV, "LTA_BENCH_TIME", "1") == "1"
const MEMORY = get(ENV, "LTA_BENCH_MEMORY", "0") == "1"      # one cell per process, measuring memory instead of time

# The CPU path of the Julia layer is what AppleAccelerate speeds up (the old benchmarks used it as well).
Sys.isapple() && METHOD == "julia_cpu" && @eval using AppleAccelerate

using LuxTriangleAttention
import LuxTriangleAttention: Lux, Static.static
using Reactant

if NEEDS_GPU
    GPU in (:metal, :cuda) || error("LTA_BENCH_GPU must be metal or cuda; got $GPU")
    if GPU === :metal
        @eval using Metal
    else
        @eval using CUDA
    end
end

include(joinpath(@__DIR__, "..", "..", "mps_client.jl"))   # `mps_client()`; a function, nothing happens on include
include(joinpath(@__DIR__, "specs.jl"))

if FRAMEWORK == "torch"
    using Test                             # test/support/parity.jl defines functions with @testset
    include(joinpath(@__DIR__, "..", "..", "python", "setup.jl"))      # `torch`, `to_py`, `to_jl`, the `sync_*!`
    include(joinpath(@__DIR__, "..", "..", "python", "alphafold3.jl"))
    include(joinpath(@__DIR__, "..", "..", "support", "parity.jl"))    # `sync_trimul!`
end

const T = Float32
const FPDIR = get(ENV, "LTA_BENCH_FPDIR", mktempdir())
const COMBOS = let s = get(ENV, "LTA_BENCH_COMBOS", "")
    isempty(s) ? nothing : Set(Tuple(split(c, "|")) for c in split(s, ";"))
end
const RTOL = parse(Float64, get(ENV, "LTA_BENCH_RTOL", "1e-3"))
const MEM_GB = parse(Float64, get(ENV, "LTA_BENCH_MEM_GB", "24"))
const SAMPLES = parse(Int, get(ENV, "LTA_BENCH_SAMPLES", "10"))
const SECONDS = parse(Float64, get(ENV, "LTA_BENCH_SECONDS", "20"))

bench_ns() = parse.(Int, split(get(ENV, "LTA_BENCH_NS", "64,128,256,384,512,768,1024"), ","; keepempty=false))

# ── the device ────────────────────────────────────────────────────────────────────────────────────

if NEEDS_GPU
    if GPU === :metal
        const DeviceArray = Metal.MtlArray
        gpu_sync() = Metal.synchronize()
        const TORCH_DEVICE = "mps"
        torch_sync() = torch.mps.synchronize()
    else
        const DeviceArray = CUDA.CuArray
        gpu_sync() = CUDA.synchronize()
        const TORCH_DEVICE = "cuda"
        torch_sync() = torch.cuda.synchronize()
    end
    # The Reactant client that traces to this GPU: jax-mps on Metal (the default client is the CPU's, under which the device
    # kernel is never reached), Reactant's own on CUDA.
    function xla_gpu_client()
        GPU === :cuda && return "gpu"
        c = mps_client()
        c === nothing && error("the jax-mps Metal PJRT plugin was not found (set JAX_MPS_PLUGIN)")
        return c
    end
end

# ── memory ────────────────────────────────────────────────────────────────────────────────────────
#
# What is measured, and why it is not the same instrument everywhere:
#
#   * CPU methods: the process's peak resident set (`Sys.maxrss()`, a high-water mark) over its baseline after loading. A fresh
#     process per cell, because a high-water mark never comes down. It includes compilation (XLA) and the inputs.
#   * GPU methods on Apple silicon: the GPU's buffers are NOT in the process RSS (a 1 GiB Metal array moved `maxrss` by nothing),
#     so RSS says nothing about them. The instrument is the Metal device's own allocation counter, `currentAllocatedSize`: it is the
#     same number Metal.jl, PyTorch (`torch.mps.driver_allocated_memory()`) and Reactant's jax-mps plugin (MLX) all move, checked
#     against each other. But the three allocators differ in WHEN the counter shows a peak:
#       - Metal.jl frees a buffer when Julia's GC finalises its array, so the counter falls again, and garbage not yet collected
#         still counts. A transient is only seen by polling while it lives.
#       - PyTorch and MLX keep freed buffers in a cache, so the counter never falls: after the run it still holds the peak. MLX
#         does not show its temporaries to a poller while the call runs (its counter jumps after the call returns); the reading
#         after the run is what holds them.
#       - PyTorch's allocator also rounds up: it takes a 1 GiB heap for a 33 MB tensor, and the counter counts the whole heap (at N=256
#         the input alone reads ~1 GiB). So for PyTorch the counter says what the process holds, not what the tensors need, and a
#         second figure is recorded: the peak of `torch.mps.current_allocated_memory()` (live tensor bytes), polled from a Python
#         thread (torch's ops release the GIL while they run). It is in the `note` column.
#     So the peak is the maximum of a poller's and of the reading after the run, taken before any GC, in a fresh process.
#     Julia methods on the GPU are measured over ONE pass: a warm-up pass (compiles; its garbage would count), a full GC, then the
#     polled pass. Earlier versions ran four passes with no GC between, which counted four passes' garbage (about 2.7x too much).
#     The other frameworks keep the four-pass reading: their caches hold the peak, so it is not garbage-sensitive.
#   * CUDA: not measured. It would need NVML's per-process figure (device-wide `used` includes other processes, and XLA
#     preallocates most of the GPU unless told not to), and I could not check that here.

if NEEDS_GPU && GPU === :metal
    device_bytes() = Int(Metal.device().currentAllocatedSize)
else
    device_bytes() = nothing
end

to_device(x::AbstractArray) = DeviceArray(x)
to_device(x::NamedTuple) = map(to_device, x)
to_device(x::Tuple) = map(to_device, x)
to_device(x) = x

# ── the output sample every method is compared on ─────────────────────────────────────────────────

# A fixed sample of at most 2^16 entries of the output, in Julia's layout: the same indices for every method, so a file of a
# few hundred KB stands for an output of hundreds of MB.
fp_indices(len) = unique(round.(Int, range(1, len; length=min(len, 1 << 16))))
fingerprint(y::AbstractArray) = Float64.(vec(y)[fp_indices(length(y))])

_slug(s) = replace(lowercase(s), r"[^a-z0-9]+" => "_")
fp_path(variant, case, method, N) = joinpath(FPDIR, "$(_slug(variant))__$(_slug(case))__$(method)__$N.f64")

save_fingerprint(path, fp) = write(path, fp)
function load_fingerprint(path)
    isfile(path) || return nothing
    return collect(reinterpret(Float64, read(path)))
end

# ── methods ───────────────────────────────────────────────────────────────────────────────────────
#
# The inputs of a cell are a tuple `inp` of arrays (or `nothing`, e.g. no mask) in the order the layer takes them before its
# parameters: `layer(inp..., ps, st)`. A `problem` is what a layer file builds once per variant:
#   layer(kernel::Bool)   the Julia layer
#   ps, st                its parameters (the same for every method, from one seed)
#   py_layer()            the PyTorch layer with the parameters copied in
#   py_inputs(inp, N)     `inp` as PyTorch takes it: `(args::Tuple, kwargs::NamedTuple)`, on the CPU

function prepare_julia(problem, inp)
    layer = problem.layer(KERNEL)
    ps, st = problem.ps, Lux.testmode(problem.st)
    if DEVICE == "gpu"
        inp, ps, st = to_device(inp), to_device(ps), to_device(st)
        sync = gpu_sync
    else
        sync = () -> nothing
    end
    out = Ref{Any}(nothing)
    run = () -> (out[] = first(layer(inp..., ps, st)); sync())
    return run, sync, () -> Array(out[])
end

# `nothing` inputs are not arguments of the compiled function: they are put back inside it.
function fill_nothings(is_nothing::Tuple, arrs::Tuple)
    i = 0
    return map(n -> n ? nothing : arrs[i+=1], is_nothing)
end

function prepare_xla(problem, inp)
    Reactant.set_default_backend(DEVICE == "gpu" ? xla_gpu_client() : "cpu")
    layer = problem.layer(KERNEL)
    is_nothing = map(isnothing, inp)
    arrs = Reactant.to_rarray(Tuple(a for a in inp if a !== nothing))
    ps_r, st_r = Reactant.to_rarray.((problem.ps, Lux.testmode(problem.st)))
    fn = (arrs, ps, st) -> first(layer(fill_nothings(is_nothing, arrs)..., ps, st))
    KERNEL && check_kernel_fired(repr(Reactant.@code_hlo optimize = false donated_args = :none fn(arrs, ps_r, st_r)))
    c = Reactant.@compile sync = true fn(arrs, ps_r, st_r)
    out = Ref{Any}(nothing)
    run = () -> (out[] = c(arrs, ps_r, st_r); nothing)
    return run, () -> nothing, () -> Array(out[])
end

# The host path of the layer gives the same numbers as the kernel, so a quiet fallback would pass every comparison and time
# the wrong thing. On Metal the kernel is a `mps.metal_kernel_lib` custom call. On CUDA I only know it is some custom call.
function check_kernel_fired(ir::String)
    marker = GPU === :metal ? "mps.metal_kernel_lib" : "custom_call"
    occursin(marker, ir) || error("the device kernel was not used (no `$marker` in the traced IR); this would time the host fallback")
    return nothing
end

py_to_device(t::Nothing) = t
py_to_device(t::AbstractVector) = [py_to_device(e) for e in t]
py_to_device(t) = t.to(TORCH_DEVICE)

function prepare_torch(problem, inp, N)
    torch.set_grad_enabled(false)
    py = problem.py_layer()
    args, kwargs = problem.py_inputs(inp, N)
    if DEVICE == "gpu"
        py = pyimport("copy").deepcopy(py).to(TORCH_DEVICE)
        args = map(py_to_device, args)
        kwargs = map(py_to_device, kwargs)
        sync = torch_sync
    else
        sync = () -> nothing
    end
    out = Ref{Any}(nothing)
    run = () -> (out[] = py(args...; kwargs...); sync())
    return run, sync, () -> Array(to_jl(out[]; swap_batch_dim=true))
end

function prepare(problem, inp, N)
    FRAMEWORK == "julia" && return prepare_julia(problem, inp)
    FRAMEWORK == "xla" && return prepare_xla(problem, inp)
    FRAMEWORK == "torch" && return prepare_torch(problem, inp, N)
    error("unknown method $METHOD")
end

# ── one cell ──────────────────────────────────────────────────────────────────────────────────────

function time_cell(run)
    b = @benchmark $run() samples = SAMPLES evals = 1 seconds = SECONDS
    return median(b).time * 1e-6, minimum(b).time * 1e-6      # ms
end

function emit(variant, case, N, status, med, mn, relerr, note="")
    @printf("RESULT\t%s\t%s\t%s\t%d\t%s\t%.6f\t%.6f\t%.6e\t%s\n", variant, case, METHOD, N, status, med, mn, relerr,
        replace(string(note), r"[\t\n\r]+" => " "))
    flush(stdout)
end

"""
    measure_memory(problem, inp, N) -> (device_peak, device_resident, rss_peak, samples, note)

Bytes above the process's baseline (after loading and a GC), for one forward pass of a freshly prepared method: the device peak (see
the header on memory above), the device bytes resident once the method is prepared (inputs, parameters, compiled buffers) and the
host's peak RSS. `device_*` are `nothing` where the device is not measured.
"""
# PyTorch's live-tensor bytes, polled by a Python thread (see the header on memory). `nothing` unless this is PyTorch on Metal.
const TORCH_POLLER_SRC = """
import threading, time, sys, torch
sys.setswitchinterval(5e-5)
class LiveBytesPoller:
    def __init__(self):
        self.peak = 0
        self.stop = False
        self.thread = threading.Thread(target=self.run, daemon=True)
    def run(self):
        while not self.stop:
            v = torch.mps.current_allocated_memory()
            if v > self.peak:
                self.peak = v
            time.sleep(0.0001)
"""

function start_torch_poller()
    (FRAMEWORK == "torch" && DEVICE == "gpu" && GPU === :metal) || return nothing
    ns = PyDict{String,PyObject}()        # a PyDict wraps the Python dict itself; a plain dict would be converted to a Julia copy
    pyimport("builtins").exec(TORCH_POLLER_SRC, ns)
    p = ns["LiveBytesPoller"]()
    p.thread.start()
    return p
end

function stop_torch_poller(p)
    p === nothing && return ""
    p.stop = true
    p.thread.join()
    return @sprintf("torch live-tensor peak %.1f MB (the device counter includes allocator heap rounding)", p.peak / 2^20)
end

function measure_memory(problem, inp, N)
    GC.gc(true)
    rss0, dev0 = Sys.maxrss(), device_bytes()
    torch_poller = start_torch_poller()
    peak, samples, stop = Ref(0), Ref(0), Ref(false)
    # The main task runs on thread 1, which is the interactive thread under `--threads=auto,1`, and a call into PyTorch or XLA
    # never yields: the poller lives in the default pool, on one of the other threads. `yield()` so it never starves libuv's timers.
    # Not under PyCall: the poller allocates, so a garbage collection can run on its thread, and PyCall's finalizers must not run
    # without the GIL (it aborts the process). PyTorch's allocator cache keeps its peak anyway, so the reading after the run holds it.
    start_poller() = dev0 === nothing || FRAMEWORK == "torch" ? nothing : Threads.@spawn(:default, while !stop[]
        peak[] = max(peak[], device_bytes()); samples[] += 1; yield()
    end)
    run, sync, _ = prepare(problem, inp, N)
    sync()
    if FRAMEWORK == "julia" && dev0 !== nothing
        # ONE forward pass. Metal.jl frees a buffer only when the GC finalises its array, and the GC does not run inside a
        # call, so a pass's peak is the sum of everything it allocates. The first run compiles and warms up (and its garbage
        # would count), so it runs unmeasured, then a full GC clears it; the polled run that follows is a single pass.
        run(); sync(); GC.gc(true)
        resident = device_bytes()
        poller = start_poller()
        run(); sync()
        after = device_bytes()                # before any GC: this pass's garbage is still there
        stop[] = true
        wait(poller)
        passpeak = max(peak[], after, resident)
        note = @sprintf("one pass after a warm-up and a GC: %.1f MB above the resident %.1f MB", (passpeak - resident) / 2^20, (resident - dev0) / 2^20)
        rss = Sys.maxrss() - rss0
        return (passpeak - dev0, resident - dev0, rss, samples[], note)
    end
    resident = device_bytes()
    poller = start_poller()
    foreach(_ -> run(), 1:4)                  # the first run compiles / warms up; the others are timed runs' worth of allocation
    sync()
    after = device_bytes()                    # before any GC: PyTorch's and MLX's caches hold the peak, Julia's garbage is still there
    stop[] = true
    poller === nothing || wait(poller)
    torch_note = stop_torch_poller(torch_poller)
    rss = Sys.maxrss() - rss0
    dev0 === nothing && return (nothing, nothing, rss, 0, torch_note)
    return (max(peak[], after, resident) - dev0, resident - dev0, rss, samples[], torch_note)
end

function memory_main(; variants, cases, make_problem, make_inputs, est_bytes)
    N = only(bench_ns())
    println("worker: method=$METHOD gpu=$(NEEDS_GPU ? GPU : :none) N=$N MEMORY threads=$(Threads.nthreads()) interactive=$(Threads.nthreads(:interactive))")
    for v in variants, c in cases
        COMBOS === nothing || (v, c) in COMBOS || continue
        println("START\t$v\t$c\t$N")
        flush(stdout)
        status, dev, res, rss, note = try
            if !KERNEL && est_bytes(N) > MEM_GB * 1e9
                ("skipped", NaN, NaN, NaN, @sprintf("estimated %.0f GB of attention logits", est_bytes(N) / 1e9))
            else
                problem = make_problem(v)
                d, r, h, n, extra = measure_memory(problem, make_inputs(c, N), N)
                note = d === nothing ? (NEEDS_GPU ? "device memory is not measured on $GPU" : "") : join(filter(!isempty, [(FRAMEWORK == "torch" ? "counter read after the run (no poller under PyCall)" : "$n counter samples"), extra]), "; ")
                ("ok", d === nothing ? NaN : d / 2^20, r === nothing ? NaN : r / 2^20, h / 2^20, note)
            end
        catch e
            ("error", NaN, NaN, NaN, first(first(split(sprint(showerror, e), '\n')), 200))
        end
        @printf("MEM\t%s\t%s\t%s\t%d\t%s\t%.3f\t%.3f\t%.3f\t%s\n", v, c, METHOD, N, status, dev, res, rss, note)
        println("DONE\t$v\t$c")
        flush(stdout)
    end
end

"""
    worker_main(; variants, cases, make_problem, make_inputs, est_bytes, kernel_dim)

Run this process's method over `variants × cases × bench_ns()`, restricted to the driver's `COMBOS`. `make_problem(variant)`,
`make_inputs(case, N) -> inp` and `est_bytes(N) -> working set of the unfused path (0 if there is no such path)` are the layer file's.
After a cell fails the larger N of that (variant, case) are skipped.
"""
function worker_main(; variants, cases, make_problem, make_inputs, est_bytes, kernel_dim)
    println("worker: method=$METHOD gpu=$(NEEDS_GPU ? GPU : :none) threads=$(Threads.nthreads()) timed=$TIMED rtol=$RTOL")
    if FRAMEWORK == "torch"
        nt = parse(Int, get(ENV, "LTA_BENCH_TORCH_THREADS", string(Sys.CPU_THREADS)))
        torch.set_num_threads(nt)
        println("worker: torch threads = ", torch.get_num_threads())
    end
    if KERNEL && kernel_dim > 0
        # tuned and cached on disk; a Reactant trace cannot tune (it would capture the device work). `kernel_dim = 0`: the
        # layer's kernel is not an attention kernel (`TriMul` compiles its own on first use, during the warm-up call).
        Base.invokelatest(LuxTriangleAttention.tune_flash_kernel!; dim=kernel_dim, T)
    end
    MEMORY && return Base.invokelatest(memory_main; variants, cases, make_problem, make_inputs, est_bytes)
    for v in variants
        problem = nothing
        for c in cases
            COMBOS === nothing || (v, c) in COMBOS || continue
            failed_at = 0
            for N in bench_ns()
                if failed_at > 0
                    emit(v, c, N, "skipped", NaN, NaN, NaN, "after failure at N=$failed_at")
                    continue
                end
                if !KERNEL && est_bytes(N) > MEM_GB * 1e9
                    emit(v, c, N, "skipped", NaN, NaN, NaN, @sprintf("estimated %.0f GB of attention logits", est_bytes(N) / 1e9))
                    failed_at = N
                    continue
                end
                println("START\t$v\t$c\t$N")
                flush(stdout)
                try
                    problem === nothing && (problem = make_problem(v))
                    run, sync, readout = prepare(problem, make_inputs(c, N), N)
                    run()                                       # compile / first call
                    y = readout()
                    all(isfinite, y) || (emit(v, c, N, "nonfinite", NaN, NaN, NaN, "$(count(!isfinite, y)) non-finite entries"); failed_at = N; continue)
                    fp = fingerprint(y)
                    save_fingerprint(fp_path(v, c, METHOD, N), fp)
                    ref = METHOD == BENCH_REF_METHOD ? nothing : load_fingerprint(fp_path(v, c, BENCH_REF_METHOD, N))
                    relerr = ref === nothing ? NaN : norm(fp .- ref) / norm(ref)
                    status = isnan(relerr) || relerr <= RTOL ? "ok" : "mismatch"
                    note = ref === nothing && METHOD != BENCH_REF_METHOD ? "unchecked: no reference at this N" : ""
                    med, mn = TIMED ? time_cell(run) : (NaN, NaN)
                    emit(v, c, N, status, med, mn, relerr, note)
                catch e
                    msg = first(split(sprint(showerror, e), '\n'))
                    emit(v, c, N, "error", NaN, NaN, NaN, first(msg, 200))
                    failed_at = N
                end
                GC.gc(true)
            end
            println("DONE\t$v\t$c")
            flush(stdout)
        end
    end
end
