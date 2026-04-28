include("../python/alphafold2.jl")

@testset "AlphaFold2" begin
    # setup
    rng = Random.Xoshiro(42)
    for T in [Float16, Float32, Float64]
        chn_in, N, B = 64, 8, 2
        chn_hidden = 32 # == D
        num_heads = 4 # == H
        inf_val = T == Float16 ? Float16(1e4) : T(1e9)

        x = randn(rng, T, chn_in, N, N, B)
        bias = randn(rng, T, num_heads, N, N, B)
        mask_cfg = (
            ("No mask", nothing),
            ("Random mask", rand(rng, Bool, N, N, B))
        )

        for (name, mask) in mask_cfg
            @testset "TriangleAttention ($name)" begin
                for is_starting in [true, false]
                    @testset "$(is_starting ? "Starting" : "Ending")" begin
                        use_bias_mha = (false, (gate=true, out=true,))
                        use_bias = (layer_norm=true, linear=false, mha=use_bias_mha,)
                        jl_layer = TriangleAttention(chn_in, chn_hidden, num_heads; is_starting, use_bias)
                        ps, st = Lux.setup(rng, jl_layer) |> convert_types(T)

                        py_layer = py"AF2TriangleAttention"(chn_in, chn_hidden, num_heads, starting=is_starting, inf=inf_val)

                        sync_triangle_attention!(py_layer, ps)

                        x_py = to_py(x; swap_batch_dim=true)
                        mask_py = isnothing(mask) ? nothing : to_py(permutedims(mask, (3, 1, 2))).to(py_dtype(T)) # Needs to be float

                        y_jl, _ = jl_layer(x, mask, ps, st)
                        py_out = py_layer(x_py, mask_py)

                        @testset "Parity ($T)" begin
                            @test y_jl ≈ to_jl(py_out; swap_batch_dim=true)
                        end

                        @testset "Type-stability ($T)" begin
                            @test_nowarn @inferred jl_layer(x, mask, ps, st)
                        end
                    end
                end
            end
        end
    end
end