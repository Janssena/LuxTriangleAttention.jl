# `Attention` against AlphaFold2's `Attention` (test/python/alphafold2.jl). The sweep is `attention_parity`
# (test/support/parity.jl); this says what is specific to the reference: its bias terms, its constructor, and that
# it takes the mask and the pair bias as two additive biases.

include("../python/alphafold2.jl")

attention_parity("af2";
    use_bias=(false, (gate=true, out=true)),
    make_py=(c, hd, H) -> py"AF2Attention"(c, c, c, hd, H),
    call_py=(py_layer, x_py, mask, mask_add_py, bias_py, T) -> py_layer(x_py, x_py; biases=[mask_add_py, bias_py]))
