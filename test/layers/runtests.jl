@testset "Python parity" begin
    include("triangle_attention.af2.jl")
    include("triangle_attention.boltz2.jl")
    include("triangle_attention.af3.jl")
end
