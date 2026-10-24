# `TriangleMultiplication` against Boltz-2's `TriangleMultiplication` (test/python/boltz2.jl). The sweep is
# `trimul_parity` (test/support/parity.jl). Boltz-2 projects `a` and `b` in one Dense (`a` first), without biases, at
# the full width of the input, and its module takes a mask always.

include("../python/boltz2.jl")

trimul_parity("boltz2";
    use_bias=false,
    chn_hidden=64,
    upcasts=true,                   # `torch.chunk(x.float(), ...)`
    make_py=(out, c, h) -> (out ? py"Boltz2TriangleMultiplicationOutgoing" : py"Boltz2TriangleMultiplicationIncoming")(c),
    sync_py=(py, ps) -> sync_trimul!(py, ps; norm_in=:norm_in, norm_out=:norm_out,
        ab=(linear=:p_in, gate=:g_in), out=(linear=:p_out, gate=:g_out)),
    mask_py=(mask, T, N, B) -> isnothing(mask) ? to_py(ones(T, N, N, B); swap_batch_dim=true) :
        to_py(permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T)))
