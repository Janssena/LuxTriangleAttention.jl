# `Attention` against AlphaFold3's `Attention` (test/python/alphafold3.jl). The sweep is `attention_parity`
# (test/support/parity.jl); this says what is specific to the reference: no bias terms, its constructor, and that it
# takes the mask and the pair bias as two additive biases.

include("../python/alphafold3.jl")

attention_parity("af3";
    use_bias=false,
    make_py=(c, hd, H) -> py"AF3Attention"(c, c, c, hd, H),
    call_py=(py_layer, x_py, mask, mask_add_py, bias_py, T) -> py_layer(x_py, x_py; biases=[mask_add_py, bias_py]))
