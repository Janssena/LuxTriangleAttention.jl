# `Attention` against Boltz-2's `Attention` (test/python/boltz2.jl). The sweep is `attention_parity`
# (test/support/parity.jl); this says what is specific to the reference: no bias terms, its constructor, and that it
# takes the pair bias and a 0/1 mask as separate arguments.

include("../python/boltz2.jl")

attention_parity("boltz2";
    use_bias=false,
    make_py=(c, hd, H) -> py"Boltz2Attention"(c, c, c, hd, H),
    call_py=(py_layer, x_py, mask, mask_add_py, bias_py, T) -> py_layer(x_py, x_py, bias_py,
        mask_add_py,
        to_py(ndims(mask) == 2 ? mask : permutedims(mask, (3, 1, 2)); swap_batch_dim=false).to(py_dtype(T))))
