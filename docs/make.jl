using Documenter
using LuxTriangleAttention

DocMeta.setdocmeta!(LuxTriangleAttention, :DocTestSetup, :(using LuxTriangleAttention); recursive=true)

makedocs(;
    modules = [LuxTriangleAttention],
    sitename = "LuxTriangleAttention.jl",
    authors = "Janssena",
    checkdocs = :exports,
    format = Documenter.HTML(;
        canonical = "https://janssena.github.io/LuxTriangleAttention.jl",
        edit_link = "main",
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = ["assets/custom.css"],
        size_threshold_warn = 300 * 2^10,   # the Internals page lists every internal docstring
        size_threshold = 600 * 2^10,
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Layers, masks and bias" => "tutorials/1-layers.md",
            "Fused attention kernels" => "tutorials/2-fused-kernels.md",
            "Compiling with Reactant" => "tutorials/3-reactant.md",
            "Half precision" => "tutorials/4-half-precision.md",
            "Your own attention kernel" => "tutorials/5-custom-kernels.md",
        ],
        "Kernel operands" => "kernel-operands.md",
        "API reference" => "api.md",
        "Internals" => "internals.md",
    ],
)

deploydocs(;
    repo = "github.com/Janssena/LuxTriangleAttention.jl.git",
    devbranch = "main",
    push_preview = true,
)
