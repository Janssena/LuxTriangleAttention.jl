# Mirrors `src/layers/`: `<layer>.test.jl` is the Julia-only tests of `src/layers/<layer>.jl`, `<layer>.<model>.test.jl` its
# Python parity against one reference (`af2`, `af3`, `boltz2`) and `<layer>.gpu.test.jl` what needs the machine's GPU. The
# layers traced by Reactant are in `test/ext/reactant/`. Each unit is tagged with its component, the backends it needs and,
# for the parity files, the model (see `test/sections.jl`).

# The parity sweeps run the native layer on the host. What is ours on other backends is compared with the native layer
# instead: `attention.gpu.test.jl`, `test/ext/reactant/attention_kernel.test.jl` and `test/ext/reactant/chunk.test.jl`.
if any(selected(c; model=m) for c in PARITY_COMPONENTS, m in MODELS)
    include("../support/parity.jl")
    @testset verbose = VERBOSE "Python parity" begin
        for model in MODELS, component in PARITY_COMPONENTS
            named(component; model) && @testset verbose = VERBOSE "$component, $model" begin
                include("$component.$model.test.jl")
            end
        end
    end
end

selected("attention") && @testset verbose = VERBOSE "attention" begin
    include("attention.test.jl")        # 16-bit: logits beyond Float16, dead rows; use_kernel/chunk_size routing; (q, k, v) tuple inputs
end

selected("attention"; backend=:gpu) && @testset verbose = VERBOSE "attention, on the GPU" begin
    include("attention.gpu.test.jl")        # use_kernel=true (each kernel) against use_kernel=false, eager
end

selected("triangle_attention") && @testset verbose = VERBOSE "triangle_attention" begin
    include("triangle_attention.test.jl")       # 16-bit: large activations through the input LayerNorm
end

selected("triangle_multiplication") && @testset verbose = VERBOSE "triangle_multiplication" begin
    include("triangle_multiplication.test.jl")  # 16-bit input LayerNorm; the planes layout
end

selected("tri_mul_core") && @testset verbose = VERBOSE "tri_mul_core" begin
    include("tri_mul_core.test.jl")             # fold_trimul, the three contraction modes, masks
end

selected("chunk") && @testset verbose = VERBOSE "chunk" begin
    include("chunk.test.jl")
end

selected("glu") && @testset verbose = VERBOSE "glu" begin
    include("glu.test.jl")
end

selected("channels_last") && @testset verbose = VERBOSE "channels_last" begin
    include("channels_last.test.jl")
end

selected("primitives") && @testset verbose = VERBOSE "primitives" begin
    include("primitives.test.jl")               # StableLayerNorm / LayerNormNoBias, Float32 statistics
end
