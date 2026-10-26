# Python references

The parity tests (`test/layers/<layer>.<model>.test.jl`) compare the Julia layers with AlphaFold2, AlphaFold3 and Boltz-2's
PyTorch code (`alphafold2.jl`, `alphafold3.jl`, `boltz2.jl`). A test run does not run Python: it reads the outputs that
were recorded from it, `test/fixtures/python/<component>.<model>.safetensors` (`test/support/fixtures.jl`). On CI and on
any machine without a local copy they come from the lazy artifact `python_fixtures` (`test/Artifacts.toml`), downloaded the
first time a parity unit runs.

## Recording them again

Needed when a reference, the sweep (`test/support/parity.jl`) or the inputs it draws change. The venv is `python/tri-attn`
(torch, einops); PyCall lives in `test/python/Project.toml` and is built against it on first use.

```
LTA_GENERATE_FIXTURES=true julia --project=test test/runtests.jl attention triangle_attention triangle_multiplication cpu
```

The sweeps then run Python for every case (`references.jl` is the Python side), compare the Julia layers with what it
returned, and write the files. To run the tests on the new files before they are published:

```
LTA_FIXTURE_DIR=$PWD/test/fixtures/python julia --project=test test/runtests.jl cpu
```

## Publishing them

The artifact is a tarball on a GitHub release. A changed set takes a new tag (an old commit's `Artifacts.toml` keeps
pointing at the tarball it was bound to):

```
julia --project=test test/python/publish_artifact.jl test-fixtures-v2
gh release create test-fixtures-v2 test/fixtures/python_fixtures.tar.gz --title test-fixtures-v2 --notes "recorded Python references"
```

The first writes `test/Artifacts.toml`; commit it and the `.safetensors` files together. Upload before pushing, as CI
downloads the tarball from the release.

## Weights

The files hold the weights too, `weights/<layout>/<path>`, which both Python and the Julia layer are run with. They are
random with no zero in them (a bias or LayerNorm bias of zero would let a layer that ignores it pass, which is what
`Lux.setup` makes of them); `random_weights` in `test/support/fixtures.jl` draws them, and the loader refuses a recorded
tensor that contains a zero. They are not the checkpoints of the models.

## When a parity test says a reference is of other inputs

Each recorded case carries a fingerprint of the inputs it was recorded with (drawn from a seeded generator, so a new Julia
can change them); record the references again.
