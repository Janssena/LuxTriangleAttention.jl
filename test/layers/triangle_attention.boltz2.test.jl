# `TriangleAttention` against Boltz-2's `TriangleAttention` (test/python/boltz2.jl). The sweep is
# `triangle_attention_parity` (test/support/parity.jl).

include("../python/boltz2.jl")

triangle_attention_parity("boltz2";
    use_bias=(false, (layer_norm=true,)),
    make_py=(c, hd, H, starting, inf) -> py"Boltz2TriangleAttention"(c, hd, H, starting=starting, inf=inf),
    sync_py = sync_triangle_attention!)
