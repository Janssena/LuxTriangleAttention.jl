# Shared plumbing for the *.bench.jl scripts: N sweep, Reactant-Metal (jax-mps)
# client setup, and result printers. Include after `using Reactant, BenchmarkTools`.

using Printf, BenchmarkTools

# N = 100:300:1000
const BENCH_NS = collect(100:300:1000)

# Uncapped @benchmark defaults (up to 10_000 samples in up to 5s, with `evals`
# auto-tuned to batch several calls into one timed sample) run our
# allocation-heavy forward passes far more times than needed for a stable
# median. At N=700-1000 each call allocates tens of MB, and on the GPU
# (Metal, unified memory) those buffers aren't reclaimed by Julia's GC until
# it decides the *CPU* heap warrants a collection — GC has no visibility into
# GPU memory pressure, so it can fill up well before that happens. Cap
# sampling explicitly rather than letting @benchmark self-tune.
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 30

# Force a full collection (and let Metal/Reactant buffer finalizers run)
# between benchmark sections — the only lever available for Metal, since
# unlike CUDA.jl there's no explicit `reclaim()`.
_gc!() = (GC.gc(true); GC.gc(true); nothing)

# Recursively wrap arrays in ConcreteRArray for Reactant; pass scalars/symbols through.
_to_rarray(x::AbstractArray) = Reactant.ConcreteRArray(x)
_to_rarray(x::NamedTuple)    = NamedTuple{keys(x)}(map(_to_rarray, values(x)))
_to_rarray(x::Tuple)         = map(_to_rarray, x)
_to_rarray(x)                = x

# Reactant-GPU (Metal) backend, via the jax-mps PJRT plugin (../jax-mps), driven
# through Reactant's generic PJRT-plugin API (see ../test-xla-stablehlo, e.g.
# src/bench_component.jl). Override the path with ENV["JAX_MPS_PLUGIN"].
const JAX_MPS_PLUGIN = get(ENV, "JAX_MPS_PLUGIN",
    joinpath(homedir(), "Postdoc", "Projects", "jax-mps", ".venv", "lib", "python3.13",
        "site-packages", "jax_plugins", "mps", "lib", "libpjrt_plugin_mps.dylib"))

# MLX (the engine behind the jax-mps plugin) reserves ~1.5x the working set by
# default and, per its own docs (../jax-mps/CLAUDE.md), does not release freed
# buffers between unrelated compiles — it expects one steady-state workload
# per process, not what we do here (sweeping many unrelated N shapes in one
# process). Left uncapped this is exactly what drove observed RSS spikes to
# 15-20GB. Must be set before the plugin creates its client, hence here,
# before `reactant_metal_client()` is ever called. Respects an existing
# setting; override with ENV["JAX_MPS_CACHE_LIMIT_BYTES"] if 4 GiB is wrong
# for your machine.
get!(ENV, "JAX_MPS_CACHE_LIMIT_BYTES", string(4 * 1024^3))

function reactant_metal_client()
    if !isfile(JAX_MPS_PLUGIN)
        @warn "jax-mps Metal PJRT plugin not found — skipping Reactant-GPU benchmarks." JAX_MPS_PLUGIN
        return nothing
    end
    try
        return Reactant.XLA.PJRT.MakeClientUsingPluginAPI(JAX_MPS_PLUGIN, "mps", "MPS")
    catch e
        @warn "Failed to create the Reactant Metal (jax-mps) client — skipping Reactant-GPU benchmarks." exception = (e, catch_backtrace())
        return nothing
    end
end

function _bprint(label, bench_jl, bench_py)
    t_jl = median(bench_jl).time
    t_py = median(bench_py).time
    @printf("    %-12s │ Julia: %10s │ PyTorch: %10s │ %6.2fx\n",
        label, BenchmarkTools.prettytime(t_jl), BenchmarkTools.prettytime(t_py), t_py / t_jl)
    flush(stdout)
end

function _bprint_xla(label, bench_jl, bench_xla)
    t_jl = median(bench_jl).time
    t_xla = median(bench_xla).time
    @printf("    %-12s │ XLA-CPU: %10s │ Julia-CPU: %10s │ %6.2fx\n",
        label, BenchmarkTools.prettytime(t_xla), BenchmarkTools.prettytime(t_jl), t_jl / t_xla)
    flush(stdout)
end

function _bprint_xla_gpu(label, bench_xla_cpu, bench_xla_gpu)
    t_cpu = median(bench_xla_cpu).time
    t_gpu = median(bench_xla_gpu).time
    @printf("    %-12s │ XLA-GPU(Metal): %10s │ XLA-CPU: %10s │ %6.2fx\n",
        label, BenchmarkTools.prettytime(t_gpu), BenchmarkTools.prettytime(t_cpu), t_cpu / t_gpu)
    flush(stdout)
end

function _bprint_gpu(label, bench_metal, bench_py_mps)
    t_metal = median(bench_metal).time
    if bench_py_mps === nothing
        @printf("    %-12s │ Metal: %10s │ Py-MPS: %10s\n",
            label, BenchmarkTools.prettytime(t_metal), "N/A")
    else
        t_mps = median(bench_py_mps).time
        @printf("    %-12s │ Metal: %10s │ Py-MPS: %10s │ %6.2fx\n",
            label, BenchmarkTools.prettytime(t_metal), BenchmarkTools.prettytime(t_mps), t_mps / t_metal)
    end
    flush(stdout)
end

# ── CSV results collection ───────────────────────────────────────────────
# Each bench script accumulates its own results, keyed by (variant, case,
# method), and calls `write_results_csv!(layer_name, results)` once at the
# end. Rows for that `layer_name` are replaced on every run; rows written by
# the *other* bench scripts are left untouched — so attention.bench.jl,
# triangle_attention.bench.jl and triangle_multiplication.bench.jl all share
# one results.bench.csv (columns = N, rows = layer × variant × case × method).
const RESULTS_CSV = joinpath(@__DIR__, "results.bench.csv")

new_results_dict() = Dict{NTuple{3,String},Dict{Int,Float64}}()

function record!(results::Dict{NTuple{3,String},Dict{Int,Float64}}, variant, case, method, N, bench)
    d = get!(results, (variant, case, method), Dict{Int,Float64}())
    d[N] = median(bench).time * 1e-9 # seconds
    return nothing
end

function _read_csv_rows(path)
    isfile(path) || return Vector{Vector{String}}()
    lines = readlines(path)
    length(lines) <= 1 && return Vector{Vector{String}}()
    return [split(l, ",") for l in lines[2:end]]
end

function write_results_csv!(layer_name::AbstractString, results::Dict{NTuple{3,String},Dict{Int,Float64}})
    header = vcat(["layer", "variant", "case", "method"], ["N=$(n)" for n in BENCH_NS])
    kept = filter(r -> r[1] != layer_name, _read_csv_rows(RESULTS_CSV))

    new_rows = map(sort(collect(keys(results)))) do key
        d = results[key]
        vcat([layer_name, key...], [haskey(d, n) ? @sprintf("%.9f", d[n]) : "" for n in BENCH_NS])
    end

    open(RESULTS_CSV, "w") do io
        println(io, join(header, ","))
        for r in kept
            println(io, join(r, ","))
        end
        for r in new_rows
            println(io, join(r, ","))
        end
    end
    @info "Wrote $(length(new_rows)) row(s) for '$layer_name' to $RESULTS_CSV (times in seconds)"
    return nothing
end

nothing
