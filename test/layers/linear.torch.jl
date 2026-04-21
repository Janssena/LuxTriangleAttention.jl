import LuxTriangleAttention: Lux

rng = Random.Xoshiro(42)
chn_in = 32
chn_hidden = 64
N = 21
batch = 10

@testset "Linear" begin
    for T in [Float16, Float32, Float64]
        @testset "$T" begin
            for use_bias = [true, false]
                @testset "use_bias = $use_bias" begin
                    py_layer = torch.nn.Linear(chn_in, chn_hidden, bias=use_bias)
                    jl_layer = Lux.Dense(chn_in => chn_hidden; use_bias)
                    ps, st = Lux.setup(rng, jl_layer) |> (T == Float64 ? Lux.f64 : (T == Float32 ? Lux.f32 : Lux.f16))

                    @test py_layer.weight.shape == size(ps.weight)
                    if use_bias
                        @test py_layer.bias.shape == size(ps.bias)
                    end

                    sync_dense!(py_layer, ps)

                    x = randn(rng, T, chn_in, N, N, batch);
                    x_py = to_py(x; swap_batch_dim=true)

                    y_jl, _ = jl_layer(x, ps, st)
                    y_py = py_layer(x_py)

                    @test y_jl ≈ to_jl(y_py; swap_batch_dim=true)
                end
            end
        end
    end
end