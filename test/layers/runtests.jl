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
