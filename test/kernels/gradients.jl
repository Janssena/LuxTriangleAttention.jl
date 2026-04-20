@testset "EnzymeRules" begin
    T = Float32
    D, H, N, B = 4, 6, 12, 1
    q = randn(rng, T, D, H, N, N, B)
    k = randn(rng, T, D, H, N, N, B)
    v = randn(rng, T, D, H, N, N, B)
    bias = randn(rng, T, H, N, N, B)

    mask_cfg = (
        ("No mask", nothing),
        ("Random mask", rand(rng, Bool, N, N, B)),
    );

    for (name, mask) in mask_cfg
        @testset "$name" begin
            out = zeros(T, size(q))
            # Enzyme will accumulate the gradients directly into these arrays.
            dq_amx = zeros(T, size(q))
            dk_amx = zeros(T, size(k))
            dv_amx = zeros(T, size(v))
            dbias_amx = zeros(T, size(bias))

            dq_tullio = zeros(T, size(q))
            dk_tullio = zeros(T, size(k))
            dv_tullio = zeros(T, size(v))
            dbias_tullio = zeros(T, size(bias))

            dq_gpu = zeros(T, size(q))
            dk_gpu = zeros(T, size(k))
            dv_gpu = zeros(T, size(v))
            dbias_gpu = zeros(T, size(bias))
            
            dout = ones(T, size(out)) 

            println("Running kernel backward functions...")
            triangle_attention_amx_backward!(dq_amx, dk_amx, dv_amx, dbias_amx, dout, copy(q), copy(k), copy(v), copy(bias), mask)
            triangle_attention_tullio_backward!(dq_tullio, dk_tullio, dv_tullio, dbias_tullio, dout, copy(q), copy(k), copy(v), copy(bias), mask)
            triangle_attention_gpu_backward!(dq_gpu, dk_gpu, dv_gpu, dbias_gpu, dout, copy(q), copy(k), copy(v), copy(bias), mask)

            function loss_fn(q_, k_, v_, bias_; _mask=mask)
                out_temp = similar(q_)
                triangle_attention_simple!(out_temp, q_, k_, v_, bias_, _mask)
                return sum(out_temp)
            end

            fdm = central_fdm(5, 1)
            println("Running Finite Differences (this may take a few seconds)...")
            grads_fd = FiniteDifferences.grad(fdm, loss_fn, q, k, v, bias)

            dq_fd = grads_fd[1]
            dk_fd = grads_fd[2]
            dv_fd = grads_fd[3]
            dbias_fd = grads_fd[4]

            @testset "Gradient parity for custom backward fuctions ($name, precision = $T)" begin
                @test dq_amx ≈ dq_fd
                @test dk_amx ≈ dk_fd
                @test dv_amx ≈ dv_fd
                @test dbias_amx ≈ dbias_fd

                @test dq_tullio ≈ dq_fd
                @test dk_tullio ≈ dk_fd
                @test dv_tullio ≈ dv_fd
                @test dbias_tullio ≈ dbias_fd

                @test dq_gpu ≈ dq_fd
                @test dk_gpu ≈ dk_fd
                @test dv_gpu ≈ dv_fd
                @test dbias_gpu ≈ dbias_fd
            end
        end
    end
end