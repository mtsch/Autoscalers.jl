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
attaching `name => new_name` pairs to the end of the macro call. These become the defaults
of the corresponding keyword arguments of [`Autoscaler`](@ref):

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
    param_mapping = Dict{Symbol,Expr}()
    for (i, p) in enumerate(params)
        param_mapping[p] = :(params[$i])
    end

    expr_left = expr.args[2]
    expr_right = expr.args[3]

    left_vars = get_variables(expr_left, [])
    :y ∉ left_vars && throw(ArgumentError("y should appear in the left-hand side"))

    right_vars = get_variables(expr_right, [])
    :x ∉ right_vars && throw(ArgumentError("x should appear in the right-hand side"))
    :y ∈ right_vars && throw(ArgumentError("y should not appear in the right-hand side"))

    expr_left = replace_params!(expr_left, param_mapping)
    expr_right = replace_params!(expr_right, param_mapping)

    scale_y = quote
        function (x, y, L, params)
            return $expr_left
        end
    end
    scale_x = quote
        function (x, L, params)
            return $expr_right
        end
    end

    remaps = Dict{Symbol,Symbol}()
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

    return quote
        ScalingAnsatz(
            $(Expr(:quote, orig_expr)),
            $scale_x, $scale_y,
            $(Tuple(params)),
            $remaps,
        )
    end
end

struct ScalingAnsatz{N,F,G}
    formula::Expr
    scale_x::F
    scale_y::G
    param_names::NTuple{N,Symbol}
    remaps::Dict{Symbol,Symbol}
end
function Base.show(io::IO, a::ScalingAnsatz)
    print(io, "@ansatz($(a.formula)")
    for (k, v) in a.remaps
        print(io, ", $k => $v")
    end
    print(io, ")")
end

const DEFAULT_ANSATZ = @ansatz(L^b * y ~ L^a * (x - x_c))
