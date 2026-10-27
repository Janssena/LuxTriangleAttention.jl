# Why do the traced (Reactant) tests fail on CUDA? The answers found with this script, for the record:
#   1. XLA's own Float32 products on Ampere run at TF32 by default (relative error 3.4e-4 against the host, 1.2e-7 at HIGHEST
#      precision): plain compiled attention, no kernel involved. The Reactant tests run at HIGHEST on CUDA for that reason.
#   2. The traced kernel was exact unless a bias was present, wrong by ~0.8 relative otherwise: the transposed pair bias is the only
#      operand XLA computes before the call, and KernelDispatch's `stablehlo.custom_call` declared no `operand_layouts`, so XLA
#      was free to hand the kernel that operand in a different physical layout. Declaring the default row-major layout for every
#      operand and result fixes every case (section 3 shows it by redefining `_embed_custom_call` for this process only).
#
#   KERNELDISPATCH_CACHE=$(mktemp -d) julia --project=test test/scratch/cuda_traced_spike.jl 2>&1 | tee lta_cuda_traced_spike.log
#
# Section 3 runs only if a case in section 2 is wrong, so once KernelDispatch declares the layouts this prints "all traced cases
# exact" and nothing else: the script doubles as the regression check for that fix.

using CUDA, Reactant, KernelDispatch, LuxTriangleAttention, Random, Printf, Statistics

const LTA = LuxTriangleAttention
const MLIR = Reactant.MLIR
Reactant.set_default_backend("gpu")

relerr(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / max(sum(abs2, Float64.(b)), eps()))

# --- 1. plain XLA ops (no kernel): compiled _sdpa_default against the host
println("--- 1. plain XLA ops (no kernel): compiled _sdpa_default against the host")
let T = Float32, hd = 8, H = 4, N = 8, B = 1
    rng = Random.Xoshiro(42)
    q, k, v = (randn(rng, T, hd, H, N, N, B) for _ in 1:3)
    bias = randn(rng, T, N, N, H, 1, B)
    mask = fill(true, N, 1, 1, N, B); mask[2, :, :, 3, :] .= false
    ref = LTA._sdpa_default(q, k, v, mask, bias)
    dev = Reactant.to_rarray
    args = (dev(q), dev(k), dev(v), dev(mask), dev(bias))
    fplain(q, k, v, m, b) = LTA._sdpa_default(q, k, v, m, b)
    plain = Array(Reactant.@jit fplain(args...))
    plain_hi = Reactant.with_config(; dot_general_precision=Reactant.PrecisionConfig.HIGHEST) do
        Array(Reactant.@jit fplain(args...))
    end
    @printf("    default precision: relative error %.3e\n    HIGHEST precision: relative error %.3e\n",
        relerr(plain, ref), relerr(plain_hi, ref))
end

# --- 2. the kernel, traced against the same kernel launched eagerly
function run_case(label; hd, H, N, hb=true, hm=true, B=1, T=Float32)
    rng = Random.Xoshiro(7)
    q, k, v = (randn(rng, T, hd, H, N, N, B) for _ in 1:3)
    bias = hb ? randn(rng, T, N, N, H, 1, B) : nothing
    mask = hm ? (m = fill(true, N, 1, 1, N, B); m[2, :, :, 3, :] .= false; m) : nothing
    LTA.tune_flash_kernel!(N; dim=hd, T)              # a trace cannot tune
    dev = Reactant.to_rarray
    d(x) = x === nothing ? nothing : dev(x)
    c(x) = x === nothing ? nothing : CuArray(x)
    eo, el = LTA.SDPA()(c(q), c(k), c(v), c(mask), c(bias))
    f(q, k, v, m, b) = LTA.SDPA()(q, k, v, m, b)
    to, tl = Reactant.@jit f(d(q), d(k), d(v), d(mask), d(bias))
    eo, el, to, tl = Array(eo), Array(el), Array(to), Array(tl)
    err = relerr(to, eo)
    # `permutation`: the error after sorting both. Near zero: the right numbers in the wrong places (a layout problem).
    @printf("  %-8s hd=%-2d H=%d N=%-2d bias=%-5s mask=%-5s  out err %.2e (permutation %.1e)  lse err %.2e\n", label, hd, H, N, hb, hm,
        err, relerr(sort(vec(to)), sort(vec(eo))), relerr(tl, el))
    return err
end

const CASES = [
    (; hd=8, H=4, N=8, hb=false, hm=false), (; hd=8, H=4, N=8, hb=false, hm=true),
    (; hd=8, H=4, N=8), (; hd=8, H=1, N=8), (; hd=16, H=4, N=8), (; hd=32, H=2, N=16), (; hd=8, H=2, N=37),
]

println("--- 2. the kernel: traced against eager")
errs = [run_case("shipped"; cs...) for cs in CASES]

# --- 3. does declaring the default layouts fix it?
if all(<(1e-4), errs)
    println("all traced cases exact")
else
    println("--- 3. the same cases with the default row-major operand_layouts / result_layouts declared (this process only)")
    # `minor_to_major` of an XLA tensor: the LAST dimension is the fastest, which is a Julia array's first (Reactant reverses
    # the dimensions), so memory is the Julia column-major order the kernels read.
    default_layouts(nds) = MLIR.IR.Attribute(
        [MLIR.IR.DenseIndexElementsAttribute(collect(Int64, (nd-1):-1:0)) for nd in nds])
    RX = Base.get_extension(KernelDispatch, :KernelDispatchReactantExt)
    Core.eval(RX, quote
        function _embed_custom_call(call_target_name::AbstractString, operands, result_shapes;
                backend_config, api_version::Integer=1, reverse=nothing,
                active_operands=nothing, label::AbstractString=call_target_name,
                result_eltypes,
                output_operand_aliases=Tuple{Int,Int}[]
        )
            rts = [
                Reactant.Ops.mlir_type(Reactant.TracedRArray{E,length(sh)}, Int64[sh...])
                for (E, sh) in zip(result_eltypes, result_shapes)
            ]
            op = Reactant.MLIR.Dialects.stablehlo.custom_call([x.mlir_data for x in operands];
                result_0=rts,
                call_target_name=Reactant.MLIR.IR.Attribute(call_target_name),
                backend_config=_config_attr(backend_config),
                api_version=Reactant.MLIR.IR.Attribute(Int32(api_version)),
                output_operand_aliases=_aliases_attr(output_operand_aliases, length(result_shapes)),
                operand_layouts=$(default_layouts)([ndims(x) for x in operands]),
                result_layouts=$(default_layouts)([length(sh) for sh in result_shapes]),
            )
            results = ntuple(length(result_shapes)) do i
                Reactant.TracedRArray{result_eltypes[i],length(result_shapes[i])}(
                    (), Reactant.MLIR.IR.result(op, i), result_shapes[i])
            end
            isnothing(reverse) ||
                _attach_reverse_rule!(op, label, operands, results, reverse, active_operands)
            return results
        end
    end)
    for cs in CASES
        run_case("patched"; cs...)
    end
end
