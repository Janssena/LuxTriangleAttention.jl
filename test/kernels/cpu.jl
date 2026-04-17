rng = Random.Xoshiro(12)

function _forward_simple(q, k, v, bias, mask)
    out = similar(q)
    triangle_attention_simple!(out, q, k, v, bias, mask)
    return out
end

function _forward_tullio(q, k, v, bias, mask)
    out = similar(q)
    triangle_attention_tullio!(out, q, k, v, bias, mask)
    return out
end

function _forward_amx(q, k, v, bias, mask)
    out = similar(q)
    triangle_attention_amx!(out, q, k, v, bias, mask)
    return out
end

for T in [Float16, Float32, Float64]
    @testset "Precision: $T" begin
        D, H, N, B = 32, 6, 16, 2
        q = randn(rng, T, D, H, N, N, B)
        k = randn(rng, T, D, H, N, N, B)
        v = randn(rng, T, D, H, N, N, B)
        bias = randn(rng, T, H, N, N, B)

        mask_cfg = (
            ("No mask", nothing),
            ("Random mask", rand(rng, Bool, N, N, B)),
            ("All-ones mask", ones(Bool, N, N, B)),
        );

        for (name, mask) in mask_cfg
            @testset "$name" begin
                y_simple = _forward_simple(q, k, v, bias, mask)
                y_tullio = _forward_tullio(q, k, v, bias, mask)
                y_amx = _forward_amx(q, k, v, bias, mask)

                @testset "Parity" begin
                    @test y_tullio ≈ y_simple
                    @test y_amx ≈ y_simple
                end

                @testset "Type-stability" begin
                    @test_nowarn @inferred _forward_amx(q, k, v, bias, mask)
                    @test_nowarn @inferred _forward_tullio(q, k, v, bias, mask)
                end
            end
        end
    end 
end

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
            dq_amx_enzyme = zeros(T, size(q))
            dk_amx_enzyme = zeros(T, size(k))
            dv_amx_enzyme = zeros(T, size(v))
            dbias_amx_enzyme = zeros(T, size(bias))

            dq_tullio_enzyme = zeros(T, size(q))
            dk_tullio_enzyme = zeros(T, size(k))
            dv_tullio_enzyme = zeros(T, size(v))
            dbias_tullio_enzyme = zeros(T, size(bias))
            
            dout = ones(T, size(out)) 

            println("Running Enzyme backward rules...")
            triangle_attention_amx_backward!(dq_amx_enzyme, dk_amx_enzyme, dv_amx_enzyme, dbias_amx_enzyme, dout, q, k, v, bias, mask)
            triangle_attention_tullio_backward!(dq_tullio_enzyme, dk_tullio_enzyme, dv_tullio_enzyme, dbias_tullio_enzyme, dout, q, k, v, bias, mask)

            function loss_fn(q_, k_, v_, bias_)
                out_temp = similar(q_)
                triangle_attention_simple!(out_temp, q_, k_, v_, bias_, mask)
                return sum(out_temp)
            end

            fdm = central_fdm(5, 1)
            println("Running Finite Differences (this may take a few seconds)...")
            grads_fd = FiniteDifferences.grad(fdm, loss_fn, q, k, v, bias)

            dq_fd = grads_fd[1]
            dk_fd = grads_fd[2]
            dv_fd = grads_fd[3]
            dbias_fd = grads_fd[4]

            @testset "Enzyme Gradient Matching ($name, precision = $T)" begin
                @test dq_amx_enzyme ≈ dq_fd
                @test dk_amx_enzyme ≈ dk_fd
                @test dv_amx_enzyme ≈ dv_fd
                @test dbias_amx_enzyme ≈ dbias_fd

                @test dq_tullio_enzyme ≈ dq_fd
                @test dk_tullio_enzyme ≈ dk_fd
                @test dv_tullio_enzyme ≈ dv_fd
                @test dbias_tullio_enzyme ≈ dbias_fd
            end
        end
    end
end