# `TriangleAttention` against AlphaFold2's `TriangleAttention` (test/python/alphafold2.jl). The sweep is
# `triangle_attention_parity` (test/support/parity.jl).

include("../python/alphafold2.jl")

triangle_attention_parity("af2";
    use_bias=(layer_norm=true, linear=false, mha=(false, (gate=true, out=true))),
    make_py=(c, hd, H, starting, inf) -> py"AF2TriangleAttention"(c, hd, H, starting=starting, inf=inf),
    sync_py = sync_triangle_attention!)
