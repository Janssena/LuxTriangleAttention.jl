using LuxTriangleAttention, Lux, Random, Test

 @testset "TriangleMultiplication Boltz2" begin
    rng = Random.default_rng()
    Random.seed!(rng, 1234)

    chn_in = 64
    hidden_chn = 32
    N = 16
    B = 2

    # --- Outgoing ---
    model = TriangleMultiplication(chn_in, hidden_chn; starting=true)
    ps, st = Lux.setup(rng, model)
    
    x = randn(Float32, chn_in, N, N, B)
    mask = rand(Bool, N, N, B)
    
    y, st_new = model(x, mask, ps, st)
    
    @test size(y) == (chn_in, N, N, B)
    @test !any(isnan, y)

    # --- Incoming ---
    model_inc = TriangleMultiplication(chn_in, hidden_chn; starting=false)
    ps_inc, st_inc = Lux.setup(rng, model_inc)
    
    y_inc, st_inc_new = model_inc(x, mask, ps_inc, st_inc)
    
    @test size(y_inc) == (chn_in, N, N, B)
    @test !any(isnan, y_inc)
end
