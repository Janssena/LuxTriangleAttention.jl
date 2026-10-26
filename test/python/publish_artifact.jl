# Packs test/fixtures/python/ into the `python_fixtures` artifact and binds it in test/Artifacts.toml.
#
#     julia --project=test test/python/publish_artifact.jl <tag>
#
# writes test/fixtures/python_fixtures.tar.gz (not tracked) and points the artifact at the GitHub release `<tag>` of
# this repository. Upload the tarball to that release under the same name, before the Artifacts.toml is pushed:
#
#     gh release create <tag> test/fixtures/python_fixtures.tar.gz --title <tag> --notes "recorded Python references"
#
# A changed set of fixtures takes a new tag: an old commit's Artifacts.toml keeps pointing at the tarball it was bound to.

import Pkg
using Pkg.Artifacts

length(ARGS) == 1 || error("usage: publish_artifact.jl <release tag>")
tag = ARGS[1]

fixtures = joinpath(@__DIR__, "..", "fixtures")
toml = joinpath(@__DIR__, "..", "Artifacts.toml")
tarball = joinpath(fixtures, "python_fixtures.tar.gz")
url = "https://github.com/Janssena/LuxTriangleAttention.jl/releases/download/$tag/python_fixtures.tar.gz"

hash = create_artifact() do dir
    for f in readdir(joinpath(fixtures, "python"); join=true)
        endswith(f, ".safetensors") && cp(f, joinpath(dir, basename(f)))
    end
end
Pkg.PlatformEngines.package(artifact_path(hash), tarball)
sha256 = bytes2hex(open(Pkg.PlatformEngines.sha256, tarball))

bind_artifact!(toml, "python_fixtures", hash; download_info=[(url, sha256)], lazy=true, force=true)
@info "bound python_fixtures" hash url tarball
