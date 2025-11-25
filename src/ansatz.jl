"""
    get_variables(expr::Expr, ignore)

Get all variable names that appear in `expr`.
"""
function get_variables(expr::Expr, ignore)
    if expr.head == :call
        vars = reduce(vcat, get_variables.(expr.args[2:end], Ref(ignore)))
    else
        throw(ArgumentError("malformed input $expr"))
    end
end
function get_variables(s::Symbol, ignore)
    if s in ignore
        return Symbol[]
    else
        return Symbol[s]
    end
end
function get_variables(r::Any, _)
    return Symbol[]
end

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

function extract_params_and_split_x_and_y(expr)
    return x_scaling, y_scaling, Tuple(params)
end

macro ansatz(expr, exprs...)
    orig_expr = deepcopy(expr)

    if expr.args[1] ≢ :~
        throw(ArgumentError("expected separator `~`, got `$(expr.args[1])`"))
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
    :x ∈ left_vars && throw(ArgumentError("x should not appear in the left-hand side"))

    right_vars = get_variables(expr_right, [])
    :x ∉ right_vars && throw(ArgumentError("x should appear in the right-hand side"))
    :y ∈ right_vars && throw(ArgumentError("y should not appear in the right-hand side"))

    expr_left = replace_params!(expr_left, param_mapping)
    expr_right = replace_params!(expr_right, param_mapping)

    # TODO is linear in y
    scale_y = quote
        function(y, L, params)
            return $expr_left
        end
    end
    scale_x = quote
        function(x, L, params)
            return $expr_right
        end
    end

    remaps = Dict{Symbol,Symbol}()
    for expr in exprs
        if expr.args[1] ≢ :(=>)
            throw(ArgumentError("extra arguments must be of the form"))
        elseif expr.args[2] ∉ (:L, :x, :y, :y_err)
            throw(ArgumentError())
        else
            remaps[expr.args[2]] = expr.args[3]
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

function relevant_columns(a::ScalingAnsatz, tbl)
    cols = Tables.columns(tbl)
    L = get(a.remaps, :L, :L)
    x = get(a.remaps, :x, :x)
    y = get(a.remaps, :y, :y)
    y_err = get(a.remaps, :y, Symbol(y, :_err))

    return (L=cols[L], x=cols[x], y=cols[y], y_err=cols[y_err])
end
