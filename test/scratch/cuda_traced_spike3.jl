# Hypothesis from test/scratch/cuda_traced_spike2.jl: the traced kernel is exact unless a bias is present, i.e. exactly when an
# operand is COMPUTED by XLA before the call (the transposed pair bias). KernelDispatch's `stablehlo.custom_call` declares no
# `operand_layouts`, so XLA's GPU layout assignment may give that operand a non-default (e.g. transposed) physical layout, which
# the kernel then reads in the wrong order.
#
# This runs the failing cases as they are, then REDEFINES `_embed_custom_call` in KernelDispatch's Reactant extension (at run
# time, in this process only) to declare the default row-major layout for every operand and result, and runs them again.
#
#   KERNELDISPATCH_CACHE=$(mktemp -d) julia --project=test test/scratch/cuda_traced_spike3.jl 2>&1 | tee lta_cuda_traced_spike3.log

using CUDA, Reactant, KernelDispatch, LuxTriangleAttention, Random, Printf

const LTA = LuxTriangleAttention
const MLIR = Reactant.MLIR
Reactant.set_default_backend("gpu")

relerr(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / max(sum(abs2, Float64.(b)), eps()))

function run_case(label; hd, H, N, hb=true, hm=true, B=1, T=Float32)
    rng = Random.Xoshiro(7)
    q, k, v = (randn(rng, T, hd, H, N, N, B) for _ in 1:3)
    bias = hb ? randn(rng, T, N, N, H, 1, B) : nothing
    mask = hm ? (m = fill(true, N, 1, 1, N, B); m[2, :, :, 3, :] .= false; m) : nothing
    LTA.tune_flash_kernel!(N; dim=hd, T)
    dev = Reactant.to_rarray
    d(x) = x === nothing ? nothing : dev(x)
    c(x) = x === nothing ? nothing : CuArray(x)
    eo, el = LTA.SDPA()(c(q), c(k), c(v), c(mask), c(bias))
    f(q, k, v, m, b) = LTA.SDPA()(q, k, v, m, b)
    to, tl = Reactant.@jit f(d(q), d(k), d(v), d(mask), d(bias))
    @printf("  %-9s hd=%-2d H=%d N=%-2d bias=%-5s mask=%-5s  out err %.2e   lse err %.2e\n", label, hd, H, N, hb, hm,
        relerr(Array(to), Array(eo)), relerr(Array(tl), Array(el)))
end

const CASES = [(; hd=8, H=4, N=8), (; hd=8, H=1, N=8), (; hd=16, H=4, N=8), (; hd=32, H=2, N=16), (; hd=8, H=2, N=37)]

println("--- as shipped")
for cs in CASES
    run_case("shipped"; cs...)
end

# The default layout of an XLA tensor, as `minor_to_major`: the LAST dimension is the fastest, which is a Julia array's
# first (Reactant reverses the dimensions), so memory is the Julia column-major order the kernels read.
default_layouts(nds) = MLIR.IR.Attribute(
    [MLIR.IR.DenseIndexElementsAttribute(collect(Int64, (nd-1):-1:0)) for nd in nds])

const RX = Base.get_extension(KernelDispatch, :KernelDispatchReactantExt)
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

println("--- with default operand_layouts / result_layouts declared")
for cs in CASES
    run_case("patched"; cs...)
end
run_case("patched"; hd=16, H=4, N=8, hb=false, hm=true)
