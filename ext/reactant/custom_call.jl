# Everything specific to writing a `stablehlo.custom_call`: the generic op-builder
# (`_embed_custom_call`), the one Metal convention built on it (`_metal_kernel_lib`), the
# hand-written-reverse-rule attachment (`_attach_reverse_rule!`), and the small helpers both need
# (`_le_bytes`, `_fresh`). None of this is specific to any one kernel — `_emit_stage` (in the main
# extension file) is the only caller today, but the seam is deliberately generic; see
# `_embed_custom_call`'s docstring.

_le_bytes(x) = "[" * join(Int.(reinterpret(UInt8, [x])), ",") * "]"

"""
    _embed_custom_call(call_target_name, operands, result_shapes; backend_config,
        api_version=1, reverse=nothing, active_operands=nothing, label=call_target_name) -> results

Emit a `stablehlo.custom_call` naming `call_target_name`, with the given `backend_config` string
attribute. The op itself carries no meaning tied to any one backend or dispatcher —
`"mps.metal_kernel_lib"` (below) is one Metal convention built on it, but a CUDA or ROCm custom
call, or a user's own, attaches through the same seam by passing a different `call_target_name`.

`backend_config` must be a StringAttr; XLA's custom-call plugins that key off JSON (as jax-mps'
does) expect it pre-serialised, not built from a NamedTuple here, since the JSON schema is entirely
the target's own convention to define.

`reverse`, given, is a VJP attached via [`_attach_reverse_rule!`](@ref) — see its docstring for the
calling convention. `label` only affects the generated reverse function's (gensym'd) name, for
readability in an MLIR dump; it defaults to `call_target_name` but is worth overriding when one
target name is shared by several logically distinct calls (`"mps.metal_kernel_lib"` dispatches
every kernel in a `.metallib` under that one target, for instance).
"""
function _embed_custom_call(call_target_name::AbstractString, operands, result_shapes;
        backend_config::AbstractString, api_version::Integer=1, reverse=nothing,
        active_operands=nothing, label::AbstractString=call_target_name)
    rts = [Reactant.Ops.mlir_type(Reactant.TracedRArray{Float32,length(sh)}, Int64[sh...])
           for sh in result_shapes]
    op = Reactant.MLIR.Dialects.stablehlo.custom_call([x.mlir_data for x in operands];
        result_0=rts,
        call_target_name=Reactant.MLIR.IR.Attribute(call_target_name),
        backend_config=Reactant.MLIR.IR.Attribute(backend_config),
        api_version=Reactant.MLIR.IR.Attribute(Int32(api_version)))
    results = ntuple(length(result_shapes)) do i
        Reactant.TracedRArray{Float32,length(result_shapes[i])}(
            (), Reactant.MLIR.IR.result(op, i), result_shapes[i])
    end
    isnothing(reverse) ||
        _attach_reverse_rule!(op, label, operands, results, reverse, active_operands)
    return results
end

"""
    _metal_kernel_lib(name, libpath, operands, result_shapes, layout; grid) -> results

[`_embed_custom_call`](@ref) targeting `"mps.metal_kernel_lib"`, jax-mps' dispatcher for a named
kernel inside a pre-compiled `.metallib`. `layout` is the kernel's argument table in order, as
`(:in, i)`, `(:out, i)`, `(:out_zero, i)` or `(:bytes, payload)`. `:out_zero` is an output jax-mps
zero-fills before dispatch (`"zero_init"`), for a kernel that accumulates into it.

Two things are easy to get wrong and are silent when wrong: the `backend_config` JSON must include
`api_version = 1`, and dispatch must be by *threadgroups*, matching how the compiled kernel indexes
by `threadgroup_position_in_grid`.
"""
function _metal_kernel_lib(name, libpath, operands, result_shapes, layout; grid,
        threadgroup=1024, reverse=nothing, active_operands=nothing)
    slots = map(enumerate(layout)) do (i, entry)
        kind, val = entry
        s = i - 1
        kind === :bytes ? "{\"slot\":$s,\"kind\":\"bytes\",\"bytes\":$val}" :
        kind === :out_zero ? "{\"slot\":$s,\"kind\":\"output\",\"arg\":$val,\"zero_init\":true}" :
        "{\"slot\":$s,\"kind\":\"$(kind === :in ? "input" : "output")\",\"arg\":$val}"
    end
    cfg = "{\"name\":\"$name\",\"metallib_path\":\"$libpath\"," *
          "\"grid\":[$(grid[1]),$(grid[2]),$(grid[3])]," *
          "\"threadgroup\":[$(threadgroup),1,1]," *
          "\"dispatch\":\"threadgroups\",\"buffers\":[" * join(slots, ",") * "]}"
          
    return _embed_custom_call("mps.metal_kernel_lib", operands, result_shapes;
        backend_config=cfg, reverse, active_operands, label=name)
end

"""
    _attach_reverse_rule!(op, name, operands, results, vjp, active_operands)

Give a `stablehlo.custom_call` a reverse-mode derivative, by tracing `vjp` into its own
`func.func` and pointing the call site at it with `enzyme.reverse`.

Enzyme cannot see inside a custom call, so the rule has to be supplied from here. It lives on the
call site rather than in a registry keyed on `call_target_name`, because one target name can
dispatch several logically distinct kernels — `"mps.metal_kernel_lib"` names whichever kernel a
call's `backend_config` JSON points at, so multiple calls sharing that target name can have
completely different adjoints. `label` in [`_embed_custom_call`](@ref) exists to keep those apart
in the generated function name.

`vjp` is called as

    vjp(primal_operands..., primal_results..., result_cotangents...)
        -> one cotangent per entry of `active_operands`, in that order

`active_operands` is 0-based, and names the operands that can carry a gradient; anything left out
is a constant, which is what the boolean mask needs. The cotangent of every result is passed even
when it is meaningless — the `lse` residual's is a zero Enzyme materialises — so the argument list
stays positional.

Enzyme verifies this signature against the call site before it fires, so a mismatch is a compile
error rather than a wrong gradient. That check is the reason this is safe to write by hand.
"""
# A new wrapper around the same MLIR value. Note `eltype` of a TracedRArray is a TracedRNumber, so
# the scalar type has to come off the type parameters rather than from `eltype`.
_fresh(x::Reactant.TracedRArray{T,N}) where {T,N} =
    Reactant.TracedRArray{T,N}((), x.mlir_data, size(x))

function _attach_reverse_rule!(op, name, operands, results, vjp, active_operands)
    # Types are what matter for the traced signature; the values are placeholders. Each slot needs
    # its OWN wrapper object, though: the tracer keys on object identity, so passing a value twice
    # -- the primal result and again as its cotangent -- collapses the two into a single block
    # argument and silently drops an argument from the signature.
    args = (_fresh.(operands)..., _fresh.(results)..., _fresh.(results)...)
    revfn = Reactant.TracedUtils.make_mlir_fn(
        vjp, args, (), string(gensym(name * "_reverse")), false;
        do_transpose=false,          # the signature has to match the call site exactly
        args_in_result=:result,      # return the cotangents only, not the arguments back
        argprefix=gensym("revarg"),
        resprefix=gensym("revres"),
        resargprefix=gensym("revresarg"),
    ).f
    # Read the symbol back rather than reusing the name we asked for: inserting into the module's
    # symbol table can uniquify it, and a stale reference here would be a dangling symbol.
    sym = String(Reactant.MLIR.IR.getattr(revfn, "sym_name"))
    Reactant.MLIR.IR.setattr!(op, "enzyme.reverse",
        Reactant.MLIR.IR.FlatSymbolRefAttribute(sym))
    Reactant.MLIR.IR.setattr!(op, "enzyme.active_operands",
        Reactant.MLIR.IR.Attribute(Int64[active_operands...]))
    return nothing
end
