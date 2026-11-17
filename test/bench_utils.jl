# The driver of the layer benchmarks (`test/{triangle_attention,triangle_multiplication}.bench.jl`). It loads no package:
# it starts one worker process per method (`test/bench/layers/<layer>.jl <method>`), reads what the worker reports and writes
# `test/results.layers.csv`.
#
# One process per method keeps the backends apart (the PJRT plugin, Metal and PyTorch each hold their own memory and do not
# give it back) and lets a cell that dies (an out-of-memory kill is not an exception) cost that cell only: the driver
# records it as `crashed` and restarts the worker on the combinations it had not reached.
#
# The first worker is the reference, PyTorch on the CPU (`BENCH_REF_METHOD`). It runs every cell once, untimed, and leaves
# a sample of each output in `LTA_BENCH_FPDIR`; every later worker compares its own output with it (`relerr`).
#
# Environment (all optional):
#   LTA_BENCH_NS         comma separated sequence lengths               (64,128,256,384,512,768,1024)
#   LTA_BENCH_METHODS    comma separated methods, replacing the default list
#   LTA_BENCH_EXTRAS=1   add the layer's extra methods (CPU and unfused-GPU ones)
#   LTA_BENCH_GPU        metal | cuda                                    (metal on a Mac, else cuda)
#   LTA_BENCH_SAMPLES    samples per cell (10);   LTA_BENCH_SECONDS: time budget per cell (20)
#   LTA_BENCH_RTOL       relative error above which a cell is `mismatch` (1e-3)
#   LTA_BENCH_MEM_GB     skip cells whose estimated working set exceeds this (24)
#   LTA_BENCH_TORCH_THREADS   torch.set_num_threads (default: all cores)
#   LTA_BENCH_NO_REF=1   skip the PyTorch CPU reference worker (no Python environment needed): `relerr` is then empty. Do not list torch methods
#   LTA_BENCH_MEMORY=1   measure memory instead of time (see `memory_driver`); LTA_BENCH_MEM_NS: its sequence lengths (128,256,512,768,1024)

using Printf, Dates

include(joinpath(@__DIR__, "bench", "layers", "specs.jl"))

const RESULTS_CSV = joinpath(@__DIR__, "results.layers.csv")
const CSV_COLUMNS = ["layer", "variant", "case", "method", "N", "status", "median_ms", "min_ms", "relerr", "note"]

bench_ns() = parse.(Int, split(get(ENV, "LTA_BENCH_NS", "64,128,256,384,512,768,1024"), ","; keepempty=false))

_clean(s) = replace(string(s), r"[,\n\r\t]+" => " ")

struct Row
    variant::String
    case::String
    method::String
    N::Int
    status::String
    median_ms::Float64
    min_ms::Float64
    relerr::Float64
    note::String
end

# The worker's stdout: `RESULT`, `START` and `DONE` lines are the protocol, the rest is passed on.
function parse_result(line)
    f = split(line, '\t')
    length(f) >= 10 || return nothing
    return Row(f[2], f[3], f[4], parse(Int, f[5]), f[6], parse(Float64, f[7]), parse(Float64, f[8]), parse(Float64, f[9]), f[10])
end

function run_worker(worker::String, method::String, combos, fpdir::String; timed::Bool)
    env = ["LTA_BENCH_FPDIR" => fpdir, "LTA_BENCH_COMBOS" => join(("$v|$c" for (v, c) in combos), ";"),
        "LTA_BENCH_TIME" => timed ? "1" : "0"]
    cmd = addenv(`$(Base.julia_cmd()) --threads=auto --project=$(dirname(Base.active_project())) $worker $method`, env...)
    errlog = joinpath(fpdir, "$(method).stderr")
    rows, done, current = Row[], Set{Tuple{String,String}}(), nothing
    proc = open(pipeline(cmd; stderr=errlog), "r")
    for line in eachline(proc)
        if startswith(line, "RESULT\t")
            r = parse_result(line)
            r === nothing || push!(rows, r)
            r === nothing || print_row(r)
        elseif startswith(line, "START\t")
            f = split(line, '\t')
            current = (f[2], f[3], parse(Int, f[4]))
        elseif startswith(line, "DONE\t")
            f = split(line, '\t')
            push!(done, (f[2], f[3]))
            current = nothing
        else
            println("    │ ", line)
        end
        flush(stdout)
    end
    wait(proc)
    return rows, done, current, proc.exitcode, errlog
end

function print_row(r::Row)
    t = isnan(r.median_ms) ? "       -" : @sprintf("%8.2f", r.median_ms)
    e = isnan(r.relerr) ? "      -" : @sprintf("%.1e", r.relerr)
    @printf("    %-24s %-10s %-18s N=%-5d %8s ms  relerr %s  %s %s\n", r.variant, r.case, r.method, r.N, t, e, r.status, r.note)
end

function bench_driver(layer::String)
    get(ENV, "LTA_BENCH_MEMORY", "0") == "1" && return memory_driver(layer)
    spec = LAYER_SPECS[layer]
    worker_files = Dict("Attention" => "attention.jl", "TriangleAttention" => "triangle_attention.jl",
        "TriangleMultiplication" => "triangle_multiplication.jl")
    worker = joinpath(@__DIR__, "bench", "layers", worker_files[layer])
    methods = haskey(ENV, "LTA_BENCH_METHODS") ? String.(split(ENV["LTA_BENCH_METHODS"], ","; keepempty=false)) :
              vcat(spec.methods, get(ENV, "LTA_BENCH_EXTRAS", "0") == "1" ? spec.extras : String[])
    ref_timed = BENCH_REF_METHOD in methods
    NO_REF = get(ENV, "LTA_BENCH_NO_REF", "0") == "1"
    fpdir = mktempdir()
    all_combos = [(v, c) for v in spec.variants for c in spec.cases]
    rows = Row[]

    println("=" ^ 100)
    println(" $layer: N = $(bench_ns())  methods = $(join(methods, ", "))")
    println(" reference: $BENCH_REF_METHOD (untimed unless listed)   gpu = ",
        get(ENV, "LTA_BENCH_GPU", Sys.isapple() ? "metal" : "cuda"))
    println("=" ^ 100)

    for method in unique(vcat(NO_REF ? String[] : [BENCH_REF_METHOD], methods))
        println("\n── $method ", "─" ^ (90 - length(method)))
        todo = copy(all_combos)
        while !isempty(todo)
            got, done, current, code, errlog = run_worker(worker, method, todo, fpdir; timed=(method != BENCH_REF_METHOD || ref_timed))
            append!(rows, got)
            todo = [cb for cb in todo if cb ∉ done]
            code == 0 && break
            # the process died inside a cell: record it and go on with the combinations it never started
            tail = isfile(errlog) ? strip(join(last(readlines(errlog), 3), " | ")) : ""
            note = "worker exit $code" * (isempty(tail) ? "" : ": " * last(tail, 160))
            if current !== nothing
                v, c, n = current
                r = Row(v, c, method, n, "crashed", NaN, NaN, NaN, _clean(note))
                push!(rows, r); print_row(r)
                todo = [cb for cb in todo if cb != (v, c)]
            else
                println("    worker failed before reaching a cell: $note")
                break
            end
        end
    end

    keep = filter(r -> r.method in methods, rows)
    write_results_csv!(layer, keep, methods)
    print_table(layer, keep, spec)
    return keep
end

# The rows of `layer` for the methods just run are replaced; the other methods' rows (an earlier, fuller run) are kept, so a
# single method can be re-run with `LTA_BENCH_METHODS`.
function write_results_csv!(layer::AbstractString, rows::Vector{Row}, methods)
    kept = String[]
    if isfile(RESULTS_CSV)
        for l in readlines(RESULTS_CSV)[2:end]
            f = split(l, ',')
            (f[1] == layer && f[4] in methods) || push!(kept, l)
        end
    end
    open(RESULTS_CSV, "w") do io
        println(io, join(CSV_COLUMNS, ","))
        foreach(l -> println(io, l), kept)
        for r in sort(rows; by=r -> (r.variant, r.case, r.method, r.N))
            println(io, join((layer, r.variant, r.case, r.method, r.N, r.status,
                    isnan(r.median_ms) ? "" : @sprintf("%.4f", r.median_ms),
                    isnan(r.min_ms) ? "" : @sprintf("%.4f", r.min_ms),
                    isnan(r.relerr) ? "" : @sprintf("%.3e", r.relerr), _clean(r.note)), ","))
        end
    end
    @info "Wrote $(length(rows)) row(s) for '$layer' to $RESULTS_CSV"
end

# median ms per method × N for each variant and case; `!` marks a mismatch with the reference, `x` a failed cell, `.` a skipped one
function print_table(layer, rows, spec)
    ns = bench_ns()
    for v in spec.variants, c in spec.cases
        sub = filter(r -> r.variant == v && r.case == c, rows)
        isempty(sub) && continue
        println("\n", layer, " — ", v, " — ", c, "   (median ms)")
        @printf("  %-18s", "method")
        foreach(n -> @printf(" %9s", "N=$n"), ns)
        println()
        for m in unique(r.method for r in sub)
            @printf("  %-18s", m)
            for n in ns
                i = findfirst(r -> r.method == m && r.N == n, sub)
                s = if i === nothing
                    "."
                else
                    r = sub[i]
                    r.status == "ok" ? @sprintf("%.2f", r.median_ms) :
                    r.status == "skipped" ? "." :
                    r.status == "mismatch" ? @sprintf("%.2f!", r.median_ms) : "x"
                end
                @printf(" %9s", s)
            end
            println()
        end
    end
end

# ── memory ────────────────────────────────────────────────────────────────────────────────────────
#
# Peak memory of one forward pass, per method and N, for the `memory` (variant, case) of the layer. Every cell is a process of its
# own (a high-water mark and an allocator's cache only ever grow, so cells cannot share one), which is why it is a separate mode
# and a smaller grid than the timing. What is measured, and the instrument for each device, is written up in
# `test/bench/layers/common.jl`: the host's peak RSS for CPU methods, the Metal device's allocation counter for GPU methods.

const MEMORY_CSV = joinpath(@__DIR__, "results.memory.csv")

struct MemRow
    variant::String
    case::String
    method::String
    N::Int
    status::String
    device_mb::Float64
    resident_mb::Float64
    rss_mb::Float64
    note::String
end

function memory_driver(layer::String)
    spec = LAYER_SPECS[layer]
    worker_files = Dict("Attention" => "attention.jl", "TriangleAttention" => "triangle_attention.jl",
        "TriangleMultiplication" => "triangle_multiplication.jl")
    worker = joinpath(@__DIR__, "bench", "layers", worker_files[layer])
    methods = haskey(ENV, "LTA_BENCH_METHODS") ? String.(split(ENV["LTA_BENCH_METHODS"], ","; keepempty=false)) :
              vcat(spec.methods, get(ENV, "LTA_BENCH_EXTRAS", "0") == "1" ? spec.extras : String[])
    ns = parse.(Int, split(get(ENV, "LTA_BENCH_MEM_NS", "128,256,512,768,1024"), ","; keepempty=false))
    variant, case = spec.memory
    logdir = mktempdir()
    rows = MemRow[]
    println("=" ^ 100)
    println(" $layer memory ($variant, $case): N = $ns  methods = $(join(methods, ", "))")
    println("=" ^ 100)
    for method in methods
        failed_at = 0
        for N in ns
            if failed_at > 0
                push!(rows, MemRow(variant, case, method, N, "skipped", NaN, NaN, NaN, "after failure at N=$failed_at"))
                continue
            end
            env = ["LTA_BENCH_MEMORY" => "1", "LTA_BENCH_NS" => string(N), "LTA_BENCH_COMBOS" => "$variant|$case",
                "LTA_BENCH_FPDIR" => logdir, "JAX_MPS_CACHE_LIMIT_BYTES" => get(ENV, "JAX_MPS_CACHE_LIMIT_BYTES", string(32 * 1024^3))]
            # `auto,1`: the main task takes the one interactive thread, so the poller of the device counter (default pool) never waits for it
            cmd = addenv(`$(Base.julia_cmd()) --threads=auto,1 --project=$(dirname(Base.active_project())) $worker $method`, env...)
            errlog = joinpath(logdir, "$method-$N.stderr")
            out = try
                read(pipeline(cmd; stderr=errlog), String)
            catch
                ""
            end
            line = findfirst(l -> startswith(l, "MEM\t"), split(out, '\n'))
            row = if line === nothing
                tail = isfile(errlog) ? strip(join(last(readlines(errlog), 3), " | ")) : ""
                MemRow(variant, case, method, N, "crashed", NaN, NaN, NaN, _clean("no result: " * last(tail, 160)))
            else
                f = split(split(out, '\n')[line], '\t')
                MemRow(f[2], f[3], f[4], parse(Int, f[5]), f[6], parse(Float64, f[7]), parse(Float64, f[8]), parse(Float64, f[9]), f[10])
            end
            push!(rows, row)
            @printf("    %-18s N=%-5d %-8s device %9s MB   resident %9s MB   rss %8s MB  %s\n", row.method, row.N, row.status,
                isnan(row.device_mb) ? "-" : @sprintf("%.1f", row.device_mb), isnan(row.resident_mb) ? "-" : @sprintf("%.1f", row.resident_mb),
                isnan(row.rss_mb) ? "-" : @sprintf("%.1f", row.rss_mb), row.note)
            flush(stdout)
            row.status == "ok" || (failed_at = N)
        end
    end
    write_memory_csv!(layer, rows, methods)
    print_memory_table(layer, rows, ns)
    return rows
end

function write_memory_csv!(layer, rows::Vector{MemRow}, methods)
    cols = ["layer", "variant", "case", "method", "N", "status", "device_mb", "resident_mb", "rss_mb", "note"]
    kept = String[]
    if isfile(MEMORY_CSV)
        for l in readlines(MEMORY_CSV)[2:end]
            f = split(l, ',')
            (f[1] == layer && f[4] in methods) || push!(kept, l)
        end
    end
    fmt(x) = isnan(x) ? "" : @sprintf("%.2f", x)
    open(MEMORY_CSV, "w") do io
        println(io, join(cols, ","))
        foreach(l -> println(io, l), kept)
        for r in rows
            println(io, join((layer, r.variant, r.case, r.method, r.N, r.status, fmt(r.device_mb), fmt(r.resident_mb), fmt(r.rss_mb), _clean(r.note)), ","))
        end
    end
    @info "Wrote $(length(rows)) row(s) for '$layer' to $MEMORY_CSV"
end

function print_memory_table(layer, rows, ns)
    for (title, get) in (("device memory above the baseline (MB): peak of one forward pass", r -> r.device_mb),
        ("host peak RSS above the baseline (MB), includes compilation", r -> r.rss_mb))
        println("\n", layer, " — ", title)
        @printf("  %-18s", "method")
        foreach(n -> @printf(" %9s", "N=$n"), ns)
        println()
        for m in unique(r.method for r in rows)
            @printf("  %-18s", m)
            for n in ns
                i = findfirst(r -> r.method == m && r.N == n, rows)
                v = i === nothing ? NaN : get(rows[i])
                @printf(" %9s", i === nothing || rows[i].status != "ok" ? (i !== nothing && rows[i].status != "skipped" ? "x" : ".") : isnan(v) ? "-" : @sprintf("%.0f", v))
            end
            println()
        end
    end
end

nothing
