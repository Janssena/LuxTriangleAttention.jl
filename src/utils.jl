"""
    resolve_defaults(defaults, keys)

Utility to resolve default values for a set of keys. This is used to handle nested 
configuration for bias and other layer options.

## Arguments
- `defaults`: The default value(s). Can be:
    - `NamedTuple`: contains specific overrides.
    - `Bool`: applied to all `keys`.
    - `Tuple{Bool, NamedTuple}`: first element is the fallback `Bool`, second is a `NamedTuple` of overrides.
- `keys`: A `NTuple{N, Symbol}` of keys to resolve defaults for.

## Returns
- A `NamedTuple` containing the resolved values for all requested keys.
"""
resolve_defaults(defaults::NamedTuple, ::NTuple{N,Symbol}) where N = defaults

resolve_defaults(default::Bool, keys::NTuple{N,Symbol}) where N =
    NamedTuple{keys}(ntuple(_ -> default, N))

function resolve_defaults(defaults::Tuple{Bool,<:NamedTuple}, _keys::NTuple{N,Symbol}) where N
    value, nt = defaults
    default_keys = filter(!∈(keys(nt)), _keys)
    defaults = NamedTuple{default_keys}(ntuple(_ -> value, length(default_keys)))
    return merge(nt, defaults)
end