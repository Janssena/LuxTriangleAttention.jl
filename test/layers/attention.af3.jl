include("../python/alphafold3.jl");

rng = Random.Xoshiro(42)

@testset "AlphaFold3" begin
    N, S, B = 16, 8, 2
    chn_in = 32
    num_heads = 6 # H
    head_dim = 8 # D / c_hidden in af3

    dims_cfg = (
        # ("3D inputs", (N, B)),
        ("4D inputs (symmetric)", (N, N, B)), # As in TriangleAttention
        # ("4D inputs (non-symmetric)", (N, S, B)), # As with pair bias
    )

    for (dim_name, _dims) in dims_cfg
        @testset "$dim_name" begin
            mask_cfg = (
                ("All-ones mask", trues(_dims...)),
                ("Random mask", rand(rng, Bool, _dims...)),
            )
            @testset "Attention" begin
                for (name, mask) in mask_cfg
                    @testset "$name" begin
                        for T in [Float16, Float32, Float64]
                            inf_val = T == Float16 ? Float16(1e4) : T(1e9)
                            x = randn(rng, T, chn_in, _dims...) # C, _dims...
                            bias = randn(rng, T, num_heads, _dims...) # H, Ni, Nj, B

                            x_jl = permutedims(x, (1, 3, 2, 4))
                            mask_jl = prep_mask(permutedims(mask, (2, 1, 3))) # Nj, 1, 1, Ni, B
                            if dim_name == "4D inputs (symmetric)"
                                bias_jl = prep_triangle_bias(permutedims(bias, (1, 3, 2, 4))) # Nj, Ni, H, 1, B
                                bias_py = to_py(permutedims(bias_jl, reverse(1:5)); swap_batch_dim=false) # B, 1, H, Ni, Nj
                            elseif dim_name == "4D inputs (non-symmetric)"
                                # TODO:
                                # bias_jl = prep_pair_bias(permutedims(bias, (1, 3, 2, 4)))
                            else
                                # TODO:
                            end

                            x_py = to_py(x; swap_batch_dim=true)
                            mask_py = permutedims(mask_jl, reverse(1:5)) # B, Ni, 1, 1, Nj
                            mask_py = to_py(inf_val .* (mask_py .- one(T)); swap_batch_dim=false)
                            
                            jl_layer = Attention(chn_in, head_dim, num_heads; use_bias=false)
                            ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                            py_layer = py"AF3Attention"(chn_in, chn_in, chn_in, head_dim, num_heads)
                            
                            sync_attention!(py_layer, ps)
                            
                            y_jl, _ = jl_layer(x_jl, bias_jl, mask_jl, ps, st)
                            
                            y_py = py_layer(x_py, x_py; biases=[mask_py, bias_py])

                            @testset "Python parity ($T)" begin
                                if ndims(x) == 4
                                    @test permutedims(y_jl, (1, 3, 2, 4)) ≈ to_jl(y_py; swap_batch_dim=true)
                                end
                            end

                            @testset "Type-stability ($T)" begin
                                @test_nowarn @inferred jl_layer(x_jl, bias_jl, mask_jl, ps, st)
                            end
                        end
                    end
                end
            end
        end
    end
end