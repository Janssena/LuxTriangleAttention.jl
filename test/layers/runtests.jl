@testset "Python parity" begin
    include("attention.af2.jl")
    include("attention.af3.jl")
    include("attention.boltz2.jl")

    include("triangle_attention.af2.jl")
    include("triangle_attention.af3.jl")
    include("triangle_attention.boltz2.jl")

    include("triangle_multiplication.af2.jl")
    include("triangle_multiplication.af3.jl")
    include("triangle_multiplication.boltz2.jl")
end

# Pure-Julia properties — no Python reference involved.
@testset "Chunking" begin
    include("attention.chunked.jl")
end

@testset "Layer routing" begin
    include("attention.routing.jl")   # use_kernel / chunk_size dispatch on Attention itself
    include("attention.tuple_inputs.jl")  # (q, k, v) tuple inputs, i.e. cross-attention
end
