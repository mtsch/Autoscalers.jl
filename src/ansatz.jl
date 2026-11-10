"""
    Ansatz(f, params; x, errors, columns=(), size=:L)

Scaling ansatz given as a function. `f(row, p)` returns the scaled `(x, y)` of one data
point, where `row` is a `NamedTuple` with the columns `size`, `x`, the keys of `errors` and
`columns`, and `p` is a `NamedTuple` of the parameters named in `params`.

Columns listed in `errors` are passed as `Measurement`s carrying the error from their error
column, so the uncertainty of the scaled `y` follows from error propagation. The scaled `y`
must depend on at least one of them and the scaled `x` on none.

* `params`: parameter names, in the order the [`Autoscaler`](@ref) expects their values.
* `x`: column with the unscaled `x`. Rows are sorted by it, and `eval_window` refers to it.
* `errors`: `NamedTuple` mapping columns to their error columns, e.g. `(; E=:dE)`.
* `columns`: further columns used by `f`.
* `size`: column with the system size. Its values are passed as integers.

# Example

```julia
Ansatz((:a, :b, :t_c); x=:t, errors=(; E=:dE)) do row, p
    return row.L^p.a * (row.t - p.t_c), row.L^p.b * row.E
end
```
"""
struct Ansatz{P,F,C<:Tuple{Vararg{Symbol}},E<:NamedTuple}
    f::F
    param_names::Tuple{Vararg{Symbol}}
    columns::C
    errors::E
    x::Symbol
    size::Symbol
    formula::Any

    # The parameter names are also a type parameter so that the `NamedTuple` of parameters
    # passed to `f` is type-stable.
    function Ansatz{P}(f::F, columns::C, errors::E, x, size, formula) where {P,F,C,E}
        return new{P,F,C,E}(f, P, columns, errors, x, size, formula)
    end
end

function Ansatz(f, params; x, errors, columns=(), size=:L, formula=nothing)
    param_names = Tuple(Symbol(p) for p in params)
    allunique(param_names) || throw(ArgumentError("parameter names must be unique"))
    x = Symbol(x)
    size = Symbol(size)
    x == size && throw(ArgumentError("`x` and `size` must be different columns"))
    if !(errors isa NamedTuple) || isempty(errors)
        throw(ArgumentError(
            "`errors` must be a non-empty `NamedTuple` mapping columns to their error " *
            "columns, e.g. `(; y=:y_err)`"
        ))
    end
    errors = map(Symbol, errors)
    for c in (x, size)
        haskey(errors, c) && throw(ArgumentError("column `$c` cannot have an error"))
    end
    cols = Tuple(unique!([size, x, Symbol.(collect(columns))..., keys(errors)...]))
    for e in errors
        e in cols && throw(ArgumentError("error column `$e` is also used as a value"))
    end
    return Ansatz{param_names}(f, cols, errors, x, size, formula)
end

function Base.show(io::IO, a::Ansatz)
    if isnothing(a.formula)
        print(io, "Ansatz(", a.f, ", ", a.param_names, "; x=", repr(a.x))
        print(io, ", errors=", a.errors, ", columns=", a.columns)
        print(io, ", size=", repr(a.size), ")")
    else
        print(io, "@ansatz(", a.formula, ")")
    end
end

"""
    parameter_tuple(a::Ansatz, params)

The values `params` as a `NamedTuple` keyed by the parameter names of `a`.
"""
function parameter_tuple(::Ansatz{P}, params) where {P}
    return NamedTuple{P}(ntuple(i -> params[i], Val(length(P))))
end

"""
    get_variables(expr, ignore)

Get all variable names that appear in `expr`, except those in `ignore`.
"""
function get_variables(expr::Expr, ignore)
    if expr.head == :call
        isempty(expr.args[2:end]) && return Symbol[]
        return reduce(vcat, get_variables.(expr.args[2:end], Ref(ignore)))
    else
        throw(ArgumentError("malformed input $expr"))
    end
end
get_variables(s::Symbol, ignore) = s in ignore ? Symbol[] : Symbol[s]
get_variables(::Any, _) = Symbol[]

"""
    replace_params!(expr::Expr, mapping::Dict{Symbol})

In `expr`, replace all occurrences of the keys of `mapping` with their corresponding values.
"""
function replace_params!(expr::Expr, mapping::Dict{Symbol})
    if expr.head == :call
        for i in 2:length(expr.args)
            a = expr.args[i]
            if a isa Symbol && haskey(mapping, a)
                expr.args[i] = mapping[a]
            elseif a isa Expr
                expr.args[i] = replace_params!(a, mapping)
            end
        end
    end
    return expr
end
replace_params!(sym, _) = sym

"""
    @ansatz expr1(y, L[, x]) ~ expr2(x, L)

A scaling ansatz to use with [`Autoscaler`](@ref), where `x` and `y` are the variables to
scale and `L` is the system size.

`expr1` and `expr2` can be arbitrary expressions. Any variables that appear in these
expressions, other than `x`, `y` and `L`, are exposed as parameters, sorted alphabetically.
For example, the ansatz `@ansatz L^b * y ~ L^a * (x - x_c)` has three parameters, `a`,`b`,
and `x_c`:

```julia
a = @ansatz L^b * y ~ L^a * (x - x_c)
a.param_names
# (:a, :b, :x_c)
```

`y` must appear on the left-hand side and `x` on the right-hand side. `x` may also appear on
the left-hand side, e.g. to subtract an analytic background:

```julia
@ansatz L^b * (y - A - B * (x - x_c)) ~ L^a * (x - x_c)
```

The column names used for `x`, `y`, `y_err`, and `L` can be remapped to match your data by
attaching `name => new_name` pairs to the end of the macro call:

```julia
a = @ansatz L^b * y ~ L^a * (x - x_c) L => system_size
```
"""
macro ansatz(expr, exprs...)
    orig_expr = deepcopy(expr)

    if !(expr isa Expr && expr.head == :call && expr.args[1] ≡ :~)
        throw(ArgumentError("expected an expression of the form `left ~ right`, got `$expr`"))
    end
    params = sort!(unique(get_variables(expr, [:x, :y, :L])))

    expr_left = expr.args[2]
    expr_right = expr.args[3]

    left_vars = get_variables(expr_left, [])
    :y ∉ left_vars && throw(ArgumentError("y should appear in the left-hand side"))

    right_vars = get_variables(expr_right, [])
    :x ∉ right_vars && throw(ArgumentError("x should appear in the right-hand side"))
    :y ∈ right_vars && throw(ArgumentError("y should not appear in the right-hand side"))

    remaps = Dict(:L => :L, :x => :x, :y => :y, :y_err => :y_err)
    for e in exprs
        if !(e isa Expr && e.head == :call && e.args[1] ≡ :(=>))
            throw(ArgumentError("extra arguments must be of the form `name => new_name`"))
        elseif e.args[2] ∉ (:L, :x, :y, :y_err)
            throw(ArgumentError(
                "only `L`, `x`, `y`, and `y_err` can be remapped, got `$(e.args[2])`"
            ))
        else
            remaps[e.args[2]] = e.args[3]
        end
    end

    mapping = Dict{Symbol,Any}(p => :(p.$p) for p in params)
    for v in (:L, :x, :y)
        mapping[v] = :(row.$(remaps[v]))
    end
    expr_left = replace_params!(expr_left, mapping)
    expr_right = replace_params!(expr_right, mapping)

    errors = :(NamedTuple{($(QuoteNode(remaps[:y])),)}(($(QuoteNode(remaps[:y_err])),)))
    return quote
        Ansatz(
            (row, p) -> ($expr_right, $expr_left),
            $(Tuple(params));
            x=$(QuoteNode(remaps[:x])),
            errors=$errors,
            size=$(QuoteNode(remaps[:L])),
            formula=$(Expr(:quote, orig_expr)),
        )
    end
end

const DEFAULT_ANSATZ = @ansatz(L^b * y ~ L^a * (x - x_c))
