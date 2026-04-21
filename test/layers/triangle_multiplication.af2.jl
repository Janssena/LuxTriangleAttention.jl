include("../python/alphafold2.jl");

function copy_weights_to_af2_triangle_multiplication_non_fused!(
    py_layer::PyObject,
    ps::NamedTuple
)   
    sync_layernorm!(py_layer.layer_norm_in, ps.layer_norm)
    sync_glu!(py_layer, ps.core.glu_a; ref=(linear=:linear_a_p, gate=:linear_a_g, ))
    sync_glu!(py_layer, ps.core.glu_b; ref=(linear=:linear_b_p, gate=:linear_b_g, ))
    sync_layernorm!(py_layer.layer_norm_out, ps.core.layer_norm_out)
    sync_glu!(py_layer, ps.core.glu_out; ref=(linear=:linear_z, gate=:linear_g, ))

    return nothing
end

rng = Random.Xoshiro(42)

@testset "TriangleMultiplication AF2" begin
    T = Float32
    chn_in = 64
    chn_hidden = 32
    N = 16
    B = 2
    x = randn(T, chn_in, N, N, B)
    
    mask_cfg = (
        ("No mask", nothing),
        ("Random mask", rand(Bool, N, N, B))
    )
    
    for is_outgoing in [true, false], (name, mask) in mask_cfg
        @testset "$(is_outgoing ? "Outgoing" : "Incoming"), $name" begin
            jl_layer = TriangleMultiplication(chn_in, chn_hidden; is_outgoing, use_bias=true, fused=false)
            ps, st = Lux.setup(rng, jl_layer)

            py_layer = py"AF2TriangleMultiplicativeUpdate"(chn_in, chn_hidden, _outgoing=is_outgoing)

            copy_weights_to_af2_triangle_multiplication_non_fused!(py_layer, ps)
            
            x_py = to_py(x; swap_batch_dim=true)
            mask_py = isnothing(mask) ? nothing : to_py(permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T))

            y_jl, st_new = jl_layer(x, mask, ps, st)
            y_py = py_layer(x_py, mask_py)

            @testset "Parity" begin
                @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
            end

            @testset "Type-stability" begin
                @test_nowarn @inferred jl_layer(x, mask, ps, st)
            end
        end
    end
end
