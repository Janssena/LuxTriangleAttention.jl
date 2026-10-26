# `TriangleMultiplication` against Boltz-2's `TriangleMultiplication` (test/python/boltz2.jl). The sweep is
# `trimul_parity` (test/support/parity.jl). Boltz-2 projects `a` and `b` in one Dense (`a` first), without biases, at
# the full width of the input, and its module takes a mask always. How Python is built is in
# test/python/references.jl.

trimul_parity("boltz2";
    use_bias=false,
    chn_hidden=64,
    upcasts=true)                   # `torch.chunk(x.float(), ...)`
