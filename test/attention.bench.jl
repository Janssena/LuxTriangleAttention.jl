# Benchmarks: Julia Attention (Lux) vs AlphaFold3 PyTorch pair-bias Attention
# (bias_layout=:kq, 3D self-attention `[C, N, B]`).
#
# Backends compared, for every N in `BENCH_NS` and with/without a mask:
#   - CPU:      Julia (+ AppleAccelerate) vs PyTorch
#   - XLA-CPU:  Reactant (CPU) vs Julia CPU
#   - XLA-GPU:  Reactant via the jax-mps Metal PJRT plugin vs XLA-CPU
#   - GPU:      Julia Metal vs PyTorch MPS
#
# Run with: julia --project=test test/attention.bench.jl
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

def to_mps(t):
    return t.to("mps")

def layer_to_mps(layer):
    import copy
    return copy.deepcopy(layer).to("mps")
"""

const rng = Random.Xoshiro(42)
const T   = Float32
const MPS_AVAILABLE  = py"torch.backends.mps.is_available()"
const GPU_CLIENT      = reactant_metal_client()

const chn_in    = 64
const head_dim  = 16
const num_heads = 8
const B         = 1

variants = (
    (name="AlphaFold3", use_bias=false, ctor=() -> py"AF3Attention"(chn_in, chn_in, chn_in, head_dim, num_heads)),
)

results = new_results_dict()

for v in variants
    println("=" ^ 92)
    println(" Attention: Julia Lux vs $(v.name) PyTorch (pair-bias, bias_layout=:kq)")
    println("=" ^ 92)
    println(" type=$(T)  C=$chn_in  D=$head_dim  H=$num_heads  B=$B  torch_threads=1  MPS_available=$(MPS_AVAILABLE)  Reactant-Metal=$(GPU_CLIENT !== nothing)")
    println()

    jl_layer = Attention(chn_in, head_dim, num_heads; use_bias=v.use_bias, bias_layout=:kq)
    ps, st = Lux.setup(rng, jl_layer)

    py_layer = v.ctor()
    sync_attention!(py_layer, ps)
    py_layer_mps = MPS_AVAILABLE ? py"layer_to_mps"(py_layer) : nothing

    f_no  = (x, b, ps, st)    -> first(jl_layer(x, b, nothing, ps, st))
    f_msk = (x, b, m, ps, st) -> first(jl_layer(x, b, m, ps, st))

    # Warm up Reactant's tracing machinery on a throwaway, tiny shape first.
    # The dominant cost (and, it turns out, memory footprint -- observed
    # peaking near 20GB) of the *first* Reactant.@compile in a session is
    # Julia JIT-compiling the whole generic tracing call graph (softmax,
    # Dense, batched_matmul, ...), not XLA itself, and that cost doesn't
    # scale with N. Paying it once here on N=4 keeps the real per-N compiles
    # below cheap (both in time and peak memory).
    let
        Nw, Bw = 4, 1
        xw, biasw, maskw = randn(rng, T, chn_in, Nw, Bw), randn(rng, T, num_heads, Nw, Nw, Bw), rand(rng, Bool, Nw, Bw)
        xw_r, biasw_r, maskw_r, psw_r, stw_r = _to_rarray.((xw, biasw, maskw, ps, Lux.testmode(st)))
        Reactant.set_default_backend("cpu")
        Reactant.@compile sync = true f_no(xw_r, biasw_r, psw_r, stw_r)
        Reactant.@compile sync = true f_msk(xw_r, biasw_r, maskw_r, psw_r, stw_r)
        _gc!()
    end

    for N in BENCH_NS
        @printf(" ── N=%-5d ───────────────────────────────────────────────\n", N)

        x    = randn(rng, T, chn_in, N, B)
        bias = randn(rng, T, num_heads, N, N, B)
        mask = rand(rng, Bool, N, B)

        bias_prep = prep_bias(bias, x, static(:kq))
        bias_py   = to_py(permutedims(bias_prep, reverse(1:ndims(bias_prep))); swap_batch_dim=false)
        x_py      = to_py(x; swap_batch_dim=true)
        mask_prep = prep_mask(mask)
        mask_bias_py = to_py(T(1e9) .* (T.(permutedims(mask_prep, reverse(1:ndims(mask_prep)))) .- one(T));
            swap_batch_dim=false).to(py_dtype(T))

        py_call_no  = () -> py_layer(x_py, x_py; biases=[bias_py])
        py_call_msk = () -> py_layer(x_py, x_py; biases=[mask_bias_py, bias_py])

        # ── CPU: Julia (+ AppleAccelerate) vs PyTorch ────────────────────────
        println("  [CPU: Julia vs PyTorch]")
        flush(stdout)
        jl_layer(x, bias, nothing, ps, st)
        jl_layer(x, bias, mask, ps, st)
        py_call_no(); py_call_msk()

        b_jl_no  = @benchmark $jl_layer($x, $bias, nothing, $ps, $st)
        b_py_no  = @benchmark $py_call_no()
        _bprint("no mask", b_jl_no, b_py_no)
        record!(results, v.name, "no mask", "Julia-CPU", N, b_jl_no)
        record!(results, v.name, "no mask", "PyTorch-CPU", N, b_py_no)

        b_jl_msk = @benchmark $jl_layer($x, $bias, $mask, $ps, $st)
        b_py_msk = @benchmark $py_call_msk()
        _bprint("with mask", b_jl_msk, b_py_msk)
        record!(results, v.name, "with mask", "Julia-CPU", N, b_jl_msk)
        record!(results, v.name, "with mask", "PyTorch-CPU", N, b_py_msk)
        _gc!()

        # ── XLA-CPU: Reactant ──────────────────────────────────────────────
        println("  [XLA-CPU: Reactant]")
        flush(stdout)
        Reactant.set_default_backend("cpu")

        x_r    = _to_rarray(x)
        bias_r = _to_rarray(bias)
        mask_r = _to_rarray(mask)
        ps_r   = _to_rarray(ps)
        st_r   = _to_rarray(Lux.testmode(st))

        t_cno  = @elapsed fn_r_no  = Reactant.@compile sync=true f_no(x_r, bias_r, ps_r, st_r)
        t_cmsk = @elapsed fn_r_msk = Reactant.@compile sync=true f_msk(x_r, bias_r, mask_r, ps_r, st_r)
        @printf("    compile: no-mask=%.1fs  with-mask=%.1fs\n", t_cno, t_cmsk)
                flush(stdout)

        b_r_no  = @benchmark $fn_r_no($x_r, $bias_r, $ps_r, $st_r)
        b_r_msk = @benchmark $fn_r_msk($x_r, $bias_r, $mask_r, $ps_r, $st_r)
        _bprint_xla("no mask", b_jl_no, b_r_no)
        _bprint_xla("with mask", b_jl_msk, b_r_msk)
        record!(results, v.name, "no mask", "Reactant-XLA-CPU", N, b_r_no)
        record!(results, v.name, "with mask", "Reactant-XLA-CPU", N, b_r_msk)
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
                bias_r_g = _to_rarray(bias)
                mask_r_g = _to_rarray(mask)
                ps_r_g   = _to_rarray(ps)
                st_r_g   = _to_rarray(Lux.testmode(st))

                t_cno_g  = @elapsed fn_r_no_g  = Reactant.@compile sync=true f_no(x_r_g, bias_r_g, ps_r_g, st_r_g)
                t_cmsk_g = @elapsed fn_r_msk_g = Reactant.@compile sync=true f_msk(x_r_g, bias_r_g, mask_r_g, ps_r_g, st_r_g)
                @printf("    compile: no-mask=%.1fs  with-mask=%.1fs\n", t_cno_g, t_cmsk_g)
                flush(stdout)

                b_r_g_no  = @benchmark $fn_r_no_g($x_r_g, $bias_r_g, $ps_r_g, $st_r_g)
                b_r_g_msk = @benchmark $fn_r_msk_g($x_r_g, $bias_r_g, $mask_r_g, $ps_r_g, $st_r_g)
                _bprint_xla_gpu("no mask", b_r_no, b_r_g_no)
                _bprint_xla_gpu("with mask", b_r_msk, b_r_g_msk)
                record!(results, v.name, "no mask", "Reactant-XLA-GPU-Metal", N, b_r_g_no)
                record!(results, v.name, "with mask", "Reactant-XLA-GPU-Metal", N, b_r_g_msk)
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
            bias_g = gdev(bias)
            mask_g = gdev(mask)
            ps_g   = gdev(ps)
            st_g   = gdev(st)

            jl_layer(x_g, bias_g, nothing, ps_g, st_g); Metal.synchronize()
            jl_layer(x_g, bias_g, mask_g, ps_g, st_g); Metal.synchronize()

            b_g_no  = @benchmark(begin
                $jl_layer($x_g, $bias_g, nothing, $ps_g, $st_g)
                Metal.synchronize()
            end, evals = 1)
            b_g_msk = @benchmark(begin
                $jl_layer($x_g, $bias_g, $mask_g, $ps_g, $st_g)
                Metal.synchronize()
            end, evals = 1)
            record!(results, v.name, "no mask", "Julia-GPU-Metal", N, b_g_no)
            record!(results, v.name, "with mask", "Julia-GPU-Metal", N, b_g_msk)

            if MPS_AVAILABLE
                x_mps         = py"to_mps"(x_py)
                bias_mps      = py"to_mps"(bias_py)
                mask_bias_mps = py"to_mps"(mask_bias_py)

                py_call_no_mps  = () -> py_layer_mps(x_mps, x_mps; biases=[bias_mps])
                py_call_msk_mps = () -> py_layer_mps(x_mps, x_mps; biases=[mask_bias_mps, bias_mps])

                py_call_no_mps(); torch.mps.synchronize()
                py_call_msk_mps(); torch.mps.synchronize()

                b_py_mps_no  = @benchmark(begin
                    $py_call_no_mps()
                    torch.mps.synchronize()
                end, evals = 1)
                b_py_mps_msk = @benchmark(begin
                    $py_call_msk_mps()
                    torch.mps.synchronize()
                end, evals = 1)
                _bprint_gpu("no mask", b_g_no, b_py_mps_no)
                _bprint_gpu("with mask", b_g_msk, b_py_mps_msk)
                record!(results, v.name, "no mask", "PyTorch-GPU-MPS", N, b_py_mps_no)
                record!(results, v.name, "with mask", "PyTorch-GPU-MPS", N, b_py_mps_msk)
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

write_results_csv!("Attention", results)
