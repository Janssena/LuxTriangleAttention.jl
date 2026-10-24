# `TriangleAttention` against AlphaFold3's `TriangleAttention` (test/python/alphafold3.jl). The sweep is
# `triangle_attention_parity` (test/support/parity.jl).

include("../python/alphafold3.jl")

triangle_attention_parity("af3";
    use_bias=(false, (layer_norm=true,)),
    make_py=(c, hd, H, starting, inf) -> py"AF3TriangleAttention"(c, hd, H, starting=starting, inf=inf),
    sync_py = sync_af3_triangle_attention!)
