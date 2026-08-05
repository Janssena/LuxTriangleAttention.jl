# Benchmarks: Julia TriangleMultiplication (Lux) vs AlphaFold3 PyTorch
# TriangleMultiplication (Algorithm 11 & 12, AlphaFold2), input `[C, N, N, B]`.
#
# Backends compared, for every N in `BENCH_NS`, Outgoing/Incoming, and with/without
# a mask:
#   - CPU:      Julia (+ AppleAccelerate) vs PyTorch
#   - XLA-CPU:  Reactant (CPU) vs Julia CPU
#   - XLA-GPU:  Reactant via the jax-mps Metal PJRT plugin vs XLA-CPU
#   - GPU:      Julia Metal vs PyTorch MPS
#
# Run with: julia --project=test test/triangle_multiplication.bench.jl
using AppleAccelerate

import Random
using Printf, LuxTriangleAttention, BenchmarkTools
import LuxTriangleAttention: Lux, Static.static
using Reactant
using Metal

include("python/setup.jl")
include("bench_utils.jl")
include("python/alphafold3.jl")

py"""
import torch
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

def to_mps(t):           return t.to("mps")
def layer_to_mps(layer):
    import copy
    return copy.deepcopy(layer).to("mps")
"""

const rng = Random.Xoshiro(42)
const T   = Float32
const MPS_AVAILABLE = py"torch.backends.mps.is_available()"
const GPU_CLIENT     = reactant_metal_client()

const py_main = pyimport("__main__")
py_class(name::AbstractString) = getproperty(py_main, Symbol(name))

const chn_in = 64
const B      = 1

# ── Python weight-sync helper (mirrors test/layers/triangle_multiplication.af3.jl) ──
function copy_weights_to_af3_triangle_multiplication!(py_layer::PyObject, ps::NamedTuple)
    sync_layernorm!(py_layer.layer_norm_in, ps.layer_norm)
    sync_glu!(py_layer, ps.core.glu_a; ref=(linear=:linear_a_p, gate=:linear_a_g))
    sync_glu!(py_layer, ps.core.glu_b; ref=(linear=:linear_b_p, gate=:linear_b_g))
    sync_layernorm!(py_layer.layer_norm_out, ps.core.layer_norm_out)
    sync_glu!(py_layer, ps.core.glu_out; ref=(linear=:linear_z, gate=:linear_g))
    return nothing
end

variant_specs = (
    (name="AlphaFold3", chn_hidden=64,
        jl_kwargs=(use_bias=true, fused=false, fused_glu=false),
        py_ctor=(ch, chh, out) -> (out ? py_class("AF3TriangleMultiplicationOutgoing") : py_class("AF3TriangleMultiplicationIncoming"))(ch, chh),
        sync=copy_weights_to_af3_triangle_multiplication!,
        py_mask_off=(N, B) -> to_py(ones(T, N, N, B); swap_batch_dim=true)),
)

results = new_results_dict()

for v in variant_specs, is_outgoing in (true, false)
    dir = is_outgoing ? "Outgoing" : "Incoming"
    variant = "$(v.name) ($dir)"
    println("=" ^ 92)
    println(" TriangleMultiplication ($dir): Julia Lux vs $(v.name) PyTorch")
    println("=" ^ 92)
    println(" type=$(T)  C=$chn_in  H=$(v.chn_hidden)  B=$B  torch_threads=1  MPS_available=$(MPS_AVAILABLE)  Reactant-Metal=$(GPU_CLIENT !== nothing)")
    println()

    jl_layer = TriangleMultiplication(chn_in, v.chn_hidden; is_outgoing, v.jl_kwargs...)
    ps, st = Lux.setup(rng, jl_layer)

    py_layer = v.py_ctor(chn_in, v.chn_hidden, is_outgoing)
    v.sync(py_layer, ps)
    py_layer_mps = MPS_AVAILABLE ? py"layer_to_mps"(py_layer) : nothing

    f_no  = (x, ps, st)    -> first(jl_layer(x, nothing, ps, st))
    f_msk = (x, m, ps, st) -> first(jl_layer(x, m, ps, st))

    # Warm up Reactant's tracing machinery on a throwaway, tiny shape first.
    # The dominant cost (and, it turns out, memory footprint -- observed
    # peaking near 20GB) of the *first* Reactant.@compile in a session is
    # Julia JIT-compiling the whole generic tracing call graph (softmax,
    # Dense, batched_matmul, ...), not XLA itself, and that cost doesn't
    # scale with N. Paying it once here on N=4 keeps the real per-N compiles
    # below cheap (both in time and peak memory).
    let
        Nw, Bw = 4, 1
        xw, maskw = randn(rng, T, chn_in, Nw, Nw, Bw), rand(rng, Bool, Nw, Nw, Bw)
        xw_r, maskw_r, psw_r, stw_r = _to_rarray.((xw, maskw, ps, Lux.testmode(st)))
        Reactant.set_default_backend("cpu")
        Reactant.@compile sync = true f_no(xw_r, psw_r, stw_r)
        Reactant.@compile sync = true f_msk(xw_r, maskw_r, psw_r, stw_r)
        _gc!()
    end

    for N in BENCH_NS
        @printf(" ── N=%-5d ───────────────────────────────────────────────\n", N)

        x    = randn(rng, T, chn_in, N, N, B)
        mask = rand(rng, Bool, N, N, B)

        x_py       = to_py(x; swap_batch_dim=true)
        mask_py    = to_py(permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T))
        mask_off_py = v.py_mask_off(N, B)

        # ── CPU: Julia (+ AppleAccelerate) vs PyTorch ────────────────────────
        println("  [CPU: Julia vs PyTorch]")
        flush(stdout)
        jl_layer(x, nothing, ps, st)
        jl_layer(x, mask, ps, st)
        py_layer(x_py, mask_off_py); py_layer(x_py, mask_py)

        b_jl_no  = @benchmark $jl_layer($x, nothing, $ps, $st)
        b_py_no  = @benchmark $py_layer($x_py, $mask_off_py)
        _bprint("no mask", b_jl_no, b_py_no)
        record!(results, variant, "no mask", "Julia-CPU", N, b_jl_no)
        record!(results, variant, "no mask", "PyTorch-CPU", N, b_py_no)

        b_jl_msk = @benchmark $jl_layer($x, $mask, $ps, $st)
        b_py_msk = @benchmark $py_layer($x_py, $mask_py)
        _bprint("with mask", b_jl_msk, b_py_msk)
        record!(results, variant, "with mask", "Julia-CPU", N, b_jl_msk)
        record!(results, variant, "with mask", "PyTorch-CPU", N, b_py_msk)
        _gc!()

        # ── XLA-CPU: Reactant ──────────────────────────────────────────────
        println("  [XLA-CPU: Reactant]")
        flush(stdout)
        Reactant.set_default_backend("cpu")

        x_r    = _to_rarray(x)
        mask_r = _to_rarray(mask)
        ps_r   = _to_rarray(ps)
        st_r   = _to_rarray(Lux.testmode(st))

        t_cno  = @elapsed fn_r_no  = Reactant.@compile sync=true f_no(x_r, ps_r, st_r)
        t_cmsk = @elapsed fn_r_msk = Reactant.@compile sync=true f_msk(x_r, mask_r, ps_r, st_r)
        @printf("    compile: no-mask=%.1fs  with-mask=%.1fs\n", t_cno, t_cmsk)
                flush(stdout)

        b_r_no  = @benchmark $fn_r_no($x_r, $ps_r, $st_r)
        b_r_msk = @benchmark $fn_r_msk($x_r, $mask_r, $ps_r, $st_r)
        _bprint_xla("no mask", b_jl_no, b_r_no)
        _bprint_xla("with mask", b_jl_msk, b_r_msk)
        record!(results, variant, "no mask", "Reactant-XLA-CPU", N, b_r_no)
        record!(results, variant, "with mask", "Reactant-XLA-CPU", N, b_r_msk)
        _gc!()

        # ── XLA-GPU: Reactant via jax-mps Metal plugin ───────────────────────
        println("  [XLA-GPU: Reactant (jax-mps Metal plugin)]")
        flush(stdout)
        if GPU_CLIENT === nothing
            println("    not available")
        else
            try
                Reactant.set_default_backend(GPU_CLIENT)

                x_r_g    = _to_rarray(x)
                mask_r_g = _to_rarray(mask)
                ps_r_g   = _to_rarray(ps)
                st_r_g   = _to_rarray(Lux.testmode(st))

                t_cno_g  = @elapsed fn_r_no_g  = Reactant.@compile sync=true f_no(x_r_g, ps_r_g, st_r_g)
                t_cmsk_g = @elapsed fn_r_msk_g = Reactant.@compile sync=true f_msk(x_r_g, mask_r_g, ps_r_g, st_r_g)
                @printf("    compile: no-mask=%.1fs  with-mask=%.1fs\n", t_cno_g, t_cmsk_g)
                flush(stdout)

                b_r_g_no  = @benchmark $fn_r_no_g($x_r_g, $ps_r_g, $st_r_g)
                b_r_g_msk = @benchmark $fn_r_msk_g($x_r_g, $mask_r_g, $ps_r_g, $st_r_g)
                _bprint_xla_gpu("no mask", b_r_no, b_r_g_no)
                _bprint_xla_gpu("with mask", b_r_msk, b_r_g_msk)
                record!(results, variant, "no mask", "Reactant-XLA-GPU-Metal", N, b_r_g_no)
                record!(results, variant, "with mask", "Reactant-XLA-GPU-Metal", N, b_r_g_msk)
            catch e
                println("    not available: $(sprint(showerror, e))")
            finally
                Reactant.set_default_backend("cpu")
                _gc!()
            end
        end

        # ── GPU: Julia Metal vs PyTorch MPS ──────────────────────────────────
        println("  [GPU: Julia Metal vs PyTorch MPS]")
        flush(stdout)
        try
            gdev   = Lux.gpu_device()
            x_g    = gdev(x)
            mask_g = gdev(mask)
            ps_g   = gdev(ps)
            st_g   = gdev(st)

            jl_layer(x_g, nothing, ps_g, st_g); Metal.synchronize()
            jl_layer(x_g, mask_g, ps_g, st_g); Metal.synchronize()

            b_g_no  = @benchmark(begin
                $jl_layer($x_g, nothing, $ps_g, $st_g)
                Metal.synchronize()
            end, evals = 1)
            b_g_msk = @benchmark(begin
                $jl_layer($x_g, $mask_g, $ps_g, $st_g)
                Metal.synchronize()
            end, evals = 1)
            record!(results, variant, "no mask", "Julia-GPU-Metal", N, b_g_no)
            record!(results, variant, "with mask", "Julia-GPU-Metal", N, b_g_msk)

            if MPS_AVAILABLE
                x_mps        = py"to_mps"(x_py)
                mask_mps     = py"to_mps"(mask_py)
                mask_off_mps = mask_off_py === nothing ? nothing : py"to_mps"(mask_off_py)

                py_layer_mps(x_mps, mask_off_mps); torch.mps.synchronize()
                py_layer_mps(x_mps, mask_mps); torch.mps.synchronize()

                b_py_mps_no  = @benchmark(begin
                    $py_layer_mps($x_mps, $mask_off_mps)
                    torch.mps.synchronize()
                end, evals = 1)
                b_py_mps_msk = @benchmark(begin
                    $py_layer_mps($x_mps, $mask_mps)
                    torch.mps.synchronize()
                end, evals = 1)
                _bprint_gpu("no mask", b_g_no, b_py_mps_no)
                _bprint_gpu("with mask", b_g_msk, b_py_mps_msk)
                record!(results, variant, "no mask", "PyTorch-GPU-MPS", N, b_py_mps_no)
                record!(results, variant, "with mask", "PyTorch-GPU-MPS", N, b_py_mps_msk)
            else
                println("    PyTorch MPS not available — skipping GPU Python benchmarks")
                _bprint_gpu("no mask", b_g_no, nothing)
                _bprint_gpu("with mask", b_g_msk, nothing)
            end
        catch e
            println("    not available: $(sprint(showerror, e))")
        end
        _gc!()

        println()
    end
end

write_results_csv!("TriangleMultiplication", results)
