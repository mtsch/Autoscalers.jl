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

[`@ansatz`](@ref) builds the same from a formula.

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
    formula::Union{Nothing,String}

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

# Collects what the expression walker finds while it rewrites one side of an `@ansatz`.
struct AnsatzContext
    params::Vector{Symbol}
    row::Symbol
    p::Symbol
    columns::Vector{Symbol}
    errors::Vector{Pair{Symbol,Symbol}}
    interpolations::Vector{Pair{Symbol,Any}}
end

function column_name(ex)
    if ex isa QuoteNode && ex.value isa Symbol
        return ex.value
    else
        throw(ArgumentError("expected a column such as `:y` in `col ± err`, got `$ex`"))
    end
end

"""
    rewrite(ex, ctx::AnsatzContext)

Rewrite one side of an `@ansatz` into the body of `f(row, p)`: columns `:c` become
`row.c`, parameters `p.name`, `\$expr` a value computed once, and every other name is
resolved in the caller's module.
"""
function rewrite(ex, ctx::AnsatzContext)
    if ex isa QuoteNode && ex.value isa Symbol
        push!(ctx.columns, ex.value)
        return Expr(:., ctx.row, QuoteNode(ex.value))
    elseif ex isa Symbol
        return ex in ctx.params ? Expr(:., ctx.p, QuoteNode(ex)) : esc(ex)
    elseif !(ex isa Expr)
        return ex
    elseif ex.head == :$
        value = gensym(:value)
        push!(ctx.interpolations, value => esc(ex.args[1]))
        return value
    elseif ex.head == :call && ex.args[1] ≡ :± && length(ex.args) == 3
        col = column_name(ex.args[2])
        push!(ctx.errors, col => column_name(ex.args[3]))
        push!(ctx.columns, col)
        return Expr(:., ctx.row, QuoteNode(col))
    elseif ex.head == :call
        f = ex.args[1]
        if f in ctx.params
            throw(ArgumentError("parameter `$f` cannot be called as a function"))
        end
        return Expr(:call, esc(f), (rewrite(a, ctx) for a in ex.args[2:end])...)
    elseif ex.head == :kw
        return Expr(:kw, ex.args[1], rewrite(ex.args[2], ctx))
    elseif ex.head == :. && ex.args[2] isa QuoteNode
        # Qualified names such as `Base.sin`: the quoted part is a field, not a column.
        return esc(ex)
    elseif ex.head == :.
        return Expr(:., esc(ex.args[1]), rewrite(ex.args[2], ctx))
    else
        return Expr(ex.head, (rewrite(a, ctx) for a in ex.args)...)
    end
end

const ANSATZ_EXAMPLE = "`@ansatz (a, b, x_c) -> :L^b * (:y ± :y_err) ~ :L^a * (:x - x_c)`"

"""
    @ansatz (params...) -> lhs ~ rhs [size=:L] [x=:col]

Define a scaling [`Ansatz`](@ref): the collapse is sought for `lhs` (the scaled `y`) as a
function of `rhs` (the scaled `x`).

* `(params...)` declares the parameters, in the order the [`Autoscaler`](@ref) expects their
  values. A single parameter can be written without parentheses.
* `:col` refers to a column of the data. Names that are not valid identifiers can be written
  as `:var"col name"`.
* `:col ± :col_err` attaches the errors in column `col_err` to `col`. The left-hand side
  must contain at least one such column; their errors propagate into the scaled `y`.
* `\$expr` inserts the value of `expr` when the ansatz is created.
* Any other name, such as a function, a constant like `π` or a variable, is resolved in the
  calling scope.
* `size` names the column with the system size (default `:L`).
* `x` names the column with the unscaled `x`, which the data are sorted by and `eval_window`
  refers to. By default it is the only column on the right-hand side besides `size`.

# Examples

```julia
@ansatz (a, b, x_c) -> :L^b * (:y ± :y_err) ~ :L^a * (:x - x_c)

# analytic background, a fixed exponent, and different column names
ν = 0.75
@ansatz (A, B, t_c) -> :N^0.1 * ((:E ± :dE) - A - B * (:t - t_c)) ~
    :N^(1/\$ν) * (:t - t_c) size=:N
```
"""
macro ansatz(lambda, options...)
    if !(lambda isa Expr && lambda.head == :->)
        throw(ArgumentError(
            "`@ansatz` expects `(params...) -> lhs ~ rhs`, e.g. $ANSATZ_EXAMPLE; got " *
            "`$lambda`. The syntax without declared parameters is no longer supported."
        ))
    end
    args, body = lambda.args
    if args isa Symbol
        params = [args]
    elseif args isa Expr && args.head == :tuple
        params = args.args
    else
        params = nothing
    end
    if isnothing(params) || !all(p -> p isa Symbol, params)
        throw(ArgumentError("parameters must be declared as `(a, b, ...) ->`, got `$args`"))
    end
    allunique(params) || throw(ArgumentError("parameter names must be unique"))
    if body isa Expr && body.head == :block
        statements = filter(a -> !(a isa LineNumberNode), body.args)
        length(statements) == 1 || throw(ArgumentError("expected a single `lhs ~ rhs`"))
        body = only(statements)
    end
    if !(body isa Expr && body.head == :call && body.args[1] ≡ :~ && length(body.args) == 3)
        throw(ArgumentError("expected `lhs ~ rhs` after `->`, got `$body`"))
    end

    size = :L
    x = nothing
    for opt in options
        if !(opt isa Expr && opt.head == :(=) && opt.args[1] in (:size, :x))
            throw(ArgumentError("unknown option `$opt`; expected `size=:col` or `x=:col`"))
        end
        value = opt.args[2]
        if value isa QuoteNode && value.value isa Symbol
            name = value.value
        elseif value isa String
            name = Symbol(value)
        else
            throw(ArgumentError("expected a column name such as `:L` in `$opt`"))
        end
        if opt.args[1] ≡ :size
            size = name
        else
            x = name
        end
    end

    ctx = AnsatzContext(params, gensym(:row), gensym(:p), Symbol[], Pair[], Pair[])
    lhs = rewrite(body.args[2], ctx)
    lhs_columns = copy(ctx.columns)
    n_errors = length(ctx.errors)
    empty!(ctx.columns)
    rhs = rewrite(body.args[3], ctx)
    rhs_columns = copy(ctx.columns)

    n_errors > 0 || throw(ArgumentError(
        "the left-hand side needs a column with errors, e.g. `(:y ± :y_err)`"
    ))
    length(ctx.errors) == n_errors ||
        throw(ArgumentError("`±` is only allowed on the left-hand side"))
    errors = Dict(ctx.errors)
    length(errors) == n_errors ||
        throw(ArgumentError("each column can have its errors attached only once"))
    for c in rhs_columns
        haskey(errors, c) && throw(ArgumentError(
            "column `$c` has errors and cannot be used on the right-hand side"
        ))
    end
    if isnothing(x)
        candidates = unique(filter(≠(size), rhs_columns))
        if length(candidates) ≠ 1
            throw(ArgumentError(
                "cannot tell which column is `x` among $(Tuple(candidates)); " *
                "specify it with e.g. `x=:$(isempty(candidates) ? :x : first(candidates))`"
            ))
        end
        x = only(candidates)
    end

    columns = Tuple(unique!([lhs_columns; rhs_columns]))
    error_names = Tuple(first.(ctx.errors))
    error_columns = Tuple(last.(ctx.errors))
    shown = string(Expr(:->, args, body))
    for opt in options
        shown *= ", " * string(opt)
    end
    construction = :($Ansatz(
        ($(ctx.row), $(ctx.p)) -> ($rhs, $lhs),
        $(Tuple(params));
        x=$(QuoteNode(x)),
        errors=NamedTuple{$error_names}($error_columns),
        columns=$columns,
        size=$(QuoteNode(size)),
        formula=$shown,
    ))
    if isempty(ctx.interpolations)
        return construction
    else
        bindings = Expr(:block, (Expr(:(=), k, v) for (k, v) in ctx.interpolations)...)
        return Expr(:let, bindings, construction)
    end
end

const DEFAULT_ANSATZ = @ansatz (a, b, x_c) -> :L^b * (:y ± :y_err) ~ :L^a * (:x - x_c)
