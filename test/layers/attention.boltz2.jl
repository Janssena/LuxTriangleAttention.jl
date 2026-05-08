include("../python/boltz2.jl");

rng = Random.Xoshiro(42)

@testset "Boltz2" begin
    N, S, B = 16, 8, 2
    chn_in = 32
    num_heads = 6 # H
    head_dim = 8 # D / c_hidden in af3

    dims_cfg = (
        ("3D inputs", :kq, (N, B)), # generic?
        ("4D inputs (symmetric)", :qk, (N, N, B)), # As in TriangleAttention
        ("4D inputs (non-symmetric)", :kq, (N, S, B)), # As in AttentionPairBias
    )

    for (dim_name, bias_layout, _dims) in dims_cfg
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
                            bias = randn(rng, T, num_heads, N, N, B) # [Ni, Nj, H, B] or [Nq, Nk, H, B]

                            if dim_name == "4D inputs (symmetric)"
                                # Swap i and j to use correct attention dim:
                                x_jl = permutedims(x, (1, 3, 2, 4))
                                # Technically we should also permute the attention dim in the mask, but 
                                # we already do this with the full reverse at y_jl ≈ permutedims(y_py, reverse(1:ndims)).
                                bias_prep = prep_bias(bias, x, static(bias_layout))
                            else
                                x_jl = x
                                bias_prep = prep_bias(bias, x, static(bias_layout))
                            end
                            
                            x_py = if dim_name == "4D inputs (non-symmetric)" 
                                # Python expects B, S, N, B in this case
                                to_py(permutedims(x, reverse(1:4)); swap_batch_dim=false)
                            else
                                to_py(x; swap_batch_dim=true)
                            end

                            mask_prep = prep_mask(mask)
                            mask_py = permutedims(mask_prep, reverse(1:ndims(mask_prep)))
                            mask_py = to_py(inf_val .* (mask_py .- one(T)); swap_batch_dim=false).to(py_dtype(T))
                            bias_py = to_py(permutedims(bias_prep, reverse(1:ndims(bias_prep))); swap_batch_dim=false)
                            
                            jl_layer = Attention(chn_in, head_dim, num_heads; use_bias=false, bias_layout)
                            ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                            py_layer = py"Boltz2Attention"(chn_in, chn_in, chn_in, head_dim, num_heads)
                            
                            sync_attention!(py_layer, ps)
                            
                            y_jl, _ = jl_layer(x_jl, bias, mask, ps, st)
                            
                            y_py = py_layer(x_py, x_py, bias_py, mask_py, to_py(ndims(mask) == 2 ? mask : permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T)))

                            @testset "Python parity ($T)" begin
                                @test y_jl ≈ permutedims(to_jl(y_py; swap_batch_dim=false), reverse(1:ndims(y_jl)))
                            end

                            @testset "Type-stability ($T)" begin
                                @test_nowarn @inferred jl_layer(x_jl, bias, mask, ps, st)
                            end
                        end
                    end
                end
            end
        end
    end
end