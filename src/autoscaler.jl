"""
    Autoscaler(
        table, ansatz::Ansatz=DEFAULT_ANSATZ;
        classic_cost_function=false,
        use_all_sizes=false,
        window_width=1,
        eval_window=nothing,
        penalty=100.0,
        fix=(;),
    )

Prepare `table` for finite-size scaling with `ansatz`. Calling the result with the free
parameters (in the order of `ansatz.param_names`, skipping hardcoded ones) returns the cost.

* `classic_cost_function`: use the cost function of
  [Houdayer & Hartmann (2004)](https://doi.org/10.1103/PhysRevB.70.014418) (per-point,
  local linear master curve) instead of the integral cost.
* `use_all_sizes`: include a curve's own size when building the master curve it is compared
  to (default: leave it out).
* `window_width`: fraction of the scaled overlap window used by the integral cost.
* `eval_window = (lo, hi)`: use the fixed-window cost. Only data points with
  `lo ≤ x ≤ hi` in the *raw* (unscaled) `x` are scored, independently of the parameters.
  Points outside it still serve as interpolation knots. Where another size does not cover a
  point's scaled `x`, its curve is extrapolated linearly and a smooth penalty `penalty * d^2`
  is added (`d` = distance outside, in units of that size's scaled range).
* `fix`: `NamedTuple` of parameters held at fixed values, e.g. `fix=(; a=1/ν)`. They are
  left out of the arguments of the cost function.

Only the columns used by `ansatz` are read. Rows with missing or non-finite values in them,
or with a non-positive error, are skipped with a warning. System sizes must be positive
integers, each size needs at least two usable rows, and no two rows may share the same size
and `x`.

The `Autoscaler` reuses internal buffers between calls, so a single instance must not be
evaluated from several threads at once.

# References

- J. Houdayer and A. K. Hartmann, *Low-temperature behavior of two-dimensional Gaussian
  Ising spin glasses*, Phys. Rev. B **70**, 014418 (2004),
  [doi:10.1103/PhysRevB.70.014418](https://doi.org/10.1103/PhysRevB.70.014418).
"""
struct Autoscaler{P,A<:Ansatz{P},D<:NamedTuple,E<:NamedTuple}
    ansatz::A

    # Columns used by the ansatz, sorted by size and `x`, and for each of them `nothing` or
    # the index of its partial derivative (see `error_dual`) and its errors.
    data::D
    errors::E

    x_val_scaled::Vector{Float64}
    y_val_scaled::Vector{Float64}
    y_err_scaled::Vector{Float64}
    sizes::Vector{Int}

    avail_sizes::Vector{Int}
    size_map::Dict{Int,UnitRange{Int}}

    # Scratch space for the classic cost function.
    selected_x_val::Vector{Float64}
    selected_y_val::Vector{Float64}
    selected_y_err::Vector{Float64}

    hardcodes::Vector{Float64}
    last_params::Vector{Float64}

    classic_cost_function::Bool
    use_all_sizes::Bool

    window_width::Float64

    eval_window::Union{Nothing,Tuple{Float64,Float64}}
    eval_mask::BitVector
    penalty::Float64
end

function Autoscaler(
    table, ansatz::Ansatz=DEFAULT_ANSATZ;
    classic_cost_function=false,
    use_all_sizes=false,
    window_width=1,
    eval_window=nothing,
    penalty=100.0,
    fix=(;),
)
    if !(fix isa NamedTuple)
        throw(ArgumentError("`fix` must be a `NamedTuple`, e.g. `(; a=1.5)`"))
    end
    for k in keys(fix)
        k in ansatz.param_names || throw(ArgumentError("`$k` in `fix` is not a parameter"))
    end
    if !isnothing(eval_window) && classic_cost_function
        throw(ArgumentError("`eval_window` and `classic_cost_function` are mutually exclusive"))
    end
    0 < window_width ≤ 1 || throw(ArgumentError("`window_width` must be in (0, 1]"))

    L = ansatz.size
    x = ansatz.x
    columns = ansatz.columns
    error_columns = Tuple(ansatz.errors)
    n_missing = n_nonfinite = n_bad_err = 0
    rows = filter(collect(Tables.rows(table))) do row
        values = map(c -> Tables.getcolumn(row, c), (columns..., error_columns...))
        if any(ismissing, values)
            n_missing += 1
        elseif !all(isfinite, values)
            n_nonfinite += 1
        elseif any(c -> Tables.getcolumn(row, c) ≤ 0, error_columns)
            n_bad_err += 1
        else
            l = Tables.getcolumn(row, L)
            if !(isinteger(l) && l > 0)
                throw(ArgumentError("system sizes `$L` must be positive integers, got $l"))
            end
            return true
        end
        return false
    end
    # Counted per call rather than using `maxlog`, which would silence the warning for
    # every later `Autoscaler` in the session.
    n_missing > 0 && @warn "Skipping $n_missing rows with missing values."
    n_nonfinite > 0 && @warn "Skipping $n_nonfinite rows with non-finite values."
    if n_bad_err > 0
        names = join(("`$c`" for c in error_columns), ", ")
        @warn "Skipping $n_bad_err rows with non-positive errors in $names."
    end
    sort!(rows; by=r -> (Tables.getcolumn(r, L), Tables.getcolumn(r, x)))

    sizes = Int[Tables.getcolumn(r, L) for r in rows]
    x_raw = Float64[Tables.getcolumn(r, x) for r in rows]
    data = NamedTuple{columns}(map(columns) do c
        c == L ? sizes : c == x ? x_raw : Float64[Tables.getcolumn(r, c) for r in rows]
    end)
    errors = NamedTuple{columns}(map(columns) do c
        if haskey(ansatz.errors, c)
            slot = findfirst(==(c), keys(ansatz.errors))
            return (slot, Float64[Tables.getcolumn(r, ansatz.errors[c]) for r in rows])
        else
            return nothing
        end
    end)

    avail_sizes = Int[]
    size_map = Dict{Int,UnitRange{Int}}()
    prev_size = -1
    start_index = 0
    for (i, curr_size) in enumerate(sizes)
        if curr_size ≠ prev_size
            if prev_size > 0
                size_map[prev_size] = start_index:(i - 1)
            end
            push!(avail_sizes, curr_size)
            prev_size = curr_size
            start_index = i
        elseif x_raw[i] == x_raw[i - 1]
            # Interpolation and the classic cost need strictly increasing x within a size.
            throw(ArgumentError("duplicate rows for L=$curr_size, `$x`=$(x_raw[i])"))
        end
    end
    isempty(avail_sizes) && throw(ArgumentError("no usable data"))
    length(avail_sizes) ≥ 2 || throw(ArgumentError("need at least two system sizes"))
    size_map[prev_size] = start_index:length(sizes)
    for l in avail_sizes
        if length(size_map[l]) < 2
            throw(ArgumentError("system size L=$l has fewer than two usable data points"))
        end
    end

    # The mask is built here, after sorting and dropping unusable rows, so its indices
    # always match the internal arrays.
    if isnothing(eval_window)
        ew = nothing
        eval_mask = trues(length(x_raw))
    else
        ew = (Float64(eval_window[1]), Float64(eval_window[2]))
        ew[1] < ew[2] || throw(ArgumentError("`eval_window` must satisfy lo < hi"))
        eval_mask = BitVector(ew[1] .≤ x_raw .≤ ew[2])
        for l in avail_sizes
            if !any(view(eval_mask, size_map[l]))
                @warn "System size L=$l has no data points inside `eval_window`; " *
                      "it will only be used for interpolation."
            end
        end
        any(eval_mask) || throw(ArgumentError("no data points inside `eval_window`"))
    end

    n = length(sizes)
    n_params = length(ansatz.param_names)
    result = Autoscaler(
        ansatz, data, errors,
        zeros(n), zeros(n), zeros(n),
        sizes, avail_sizes, size_map,
        Float64[], Float64[], Float64[],
        fill(NaN, n_params), zeros(n_params),
        classic_cost_function, use_all_sizes, Float64(window_width),
        ew, eval_mask, Float64(penalty),
    )
    return hardcode_params!(result, fix)
end

function hardcode_params!(a::Autoscaler, fix::NamedTuple)
    for (i, p) in enumerate(a.ansatz.param_names)
        a.hardcodes[i] = get(fix, p, NaN)
    end
    return a
end

Base.show(io::IO, a::Autoscaler) = print(io, "Autoscaler(", a.ansatz, ")")

"""
    free_parameter_names(a::Autoscaler)

Names of the parameters that are not hardcoded, in the order the cost function expects them.
"""
function free_parameter_names(a::Autoscaler)
    return Tuple(p for (p, h) in zip(a.ansatz.param_names, a.hardcodes) if isnan(h))
end

# Tags the dual numbers that carry the data errors, as `ForwardDiff.Tag{ErrorPropagation}`,
# so they cannot be confused with duals of other differentiations.
struct ErrorPropagation end
const ErrorTag = ForwardDiff.Tag{ErrorPropagation}

"""
    error_dual(tag, value, (slot, errors), i, ::Val{K})

`value` as a dual number whose `slot`-th of `K` partials is its error. Propagated through
the ansatz, the partials of the scaled `y` are then `∂y/∂c * σ_c` for each column `c` with
errors, and their norm is the first-order error of the scaled `y`.
"""
function error_dual(::T, value, (slot, errors), i, ::Val{K}) where {T,K}
    return Dual{T}(value, ntuple(j -> j == slot ? errors[i] : 0.0, Val(K)))
end

function transform!(a::Autoscaler, params)
    p = parameter_tuple(a.ansatz, params)
    f = a.ansatz.f
    n_errors = Val(length(a.ansatz.errors))
    # ForwardDiff nests duals by the order in which their tags were first created, and the
    # tag of the innermost differentiation must be the outermost dual. Creating the tag
    # here, from the parameter type, puts it after the tags of any differentiation of the
    # cost (whose duals are the parameters) and before any tag the ansatz creates.
    tag = ForwardDiff.Tag(ErrorPropagation(), eltype(params))
    for i in eachindex(a.x_val_scaled)
        row = map(a.data, a.errors) do values, errors
            if isnothing(errors)
                return values[i]
            else
                return error_dual(tag, values[i], errors, i, n_errors)
            end
        end
        x, y = f(row, p)
        a.x_val_scaled[i] = scaled_x(x)
        a.y_val_scaled[i], a.y_err_scaled[i] = scaled_y(y)
    end
    # Interpolation and the overlap window assume ascending scaled `x` within each size.
    for size in a.avail_sizes
        if !issorted(view(a.x_val_scaled, a.size_map[size]))
            throw(ArgumentError(
                "scaled `x` is not ascending for L=$size at parameters $params; the " *
                "ansatz must map `x` to an increasing function of `x` for every size"
            ))
        end
    end
    return nothing
end

scaled_x(x::Real) = Float64(x)
function scaled_x(::Dual{<:ErrorTag})
    throw(ArgumentError("the scaled `x` must not depend on columns with errors"))
end

function scaled_y(y::Dual{<:ErrorTag})
    # `hypot` rescales, so tiny or huge errors do not underflow or overflow when squared.
    return ForwardDiff.value(y), hypot(ForwardDiff.partials(y).values...)
end
function scaled_y(::Real)
    throw(ArgumentError(
        "the scaled `y` has no error; it must depend on a column listed in `errors`"
    ))
end

"""
    raw_x(a::Autoscaler)

Unscaled `x` of all data points, in the internal order.
"""
raw_x(a::Autoscaler) = a.data[a.ansatz.x]

function scaled_data(a::Autoscaler, size)
    range = a.size_map[size]
    return (
        x_val=view(a.x_val_scaled, range),
        y_val=view(a.y_val_scaled, range),
        y_err=view(a.y_err_scaled, range),
    )
end

"""
    window(a::Autoscaler)

Overlap of the scaled `x` ranges of all sizes, shrunk to a fraction `a.window_width`
(shrinking the side further from zero first). Used by the integral cost function.
"""
function window(a::Autoscaler)
    lo = -Inf
    hi = Inf
    for size in a.avail_sizes
        x_scaled = scaled_data(a, size).x_val
        lo = max(lo, first(x_scaled))
        hi = min(hi, last(x_scaled))
    end
    if lo > hi
        return lo, hi
    end
    if a.window_width ≠ 1
        width = hi - lo
        to_shrink = width * (1 - a.window_width)

        diff = abs(hi) - abs(lo)
        if diff < 0
            step = min(-diff, to_shrink)
            lo += step
        else
            step = min(diff, to_shrink)
            hi -= step
        end
        to_shrink -= step

        lo += to_shrink / 2
        hi -= to_shrink / 2
    end
    return lo, hi
end

"""
    get_parameters(a::Autoscaler, args)

Combine the free parameters `args` with the hardcoded ones into the full parameter vector.
"""
get_parameters(a::Autoscaler, args) = copy(get_parameters!(a, args))

"""
    get_parameters!(a::Autoscaler, args)

Like [`get_parameters`](@ref), but returns an internal buffer that is overwritten on the
next call. Avoids an allocation per cost evaluation.
"""
function get_parameters!(a::Autoscaler, args)
    expected = count(isnan, a.hardcodes)
    if length(args) ≠ expected
        throw(ArgumentError("expected $(expected) params, got $(length(args))"))
    end

    params = a.last_params
    arg_idx = 1
    for i in eachindex(params)
        if isnan(a.hardcodes[i])
            params[i] = args[arg_idx]
            arg_idx += 1
        else
            params[i] = a.hardcodes[i]
        end
    end
    return params
end

(a::Autoscaler)(params::AbstractVector) = cost_function(a, get_parameters!(a, params))
(a::Autoscaler)(args::Vararg{Real}) = cost_function(a, get_parameters!(a, args))

"""
    to_table(a::Autoscaler, args...)

Scale the data with the free parameters `args` and return a named tuple of columns
`L`, `x`, `y`, `y_err` (scaled), `master_curve` (value of the master curve the point is
compared to), and `in_window` (whether the point is scored by the fixed-window cost).
"""
to_table(a::Autoscaler, args::Vararg{Real}) = to_table(a, args)
function to_table(a::Autoscaler, args::Union{AbstractVector,Tuple})
    params = get_parameters!(a, args)
    transform!(a, params)
    if !isnothing(a.eval_window)
        master = [fixed_master_curve(a, x, L).Y for (x, L) in zip(a.x_val_scaled, a.sizes)]
    elseif a.classic_cost_function
        master = map(zip(a.x_val_scaled, a.sizes)) do (x, L)
            m = standard_master_curve(a, x, L)
            ismissing(m) ? NaN : m[1]
        end
    else
        master = [first(new_master_curve(a, x, L)) for (x, L) in zip(a.x_val_scaled, a.sizes)]
    end
    return (;
        L=copy(a.sizes),
        x=copy(a.x_val_scaled),
        y=copy(a.y_val_scaled),
        y_err=copy(a.y_err_scaled),
        master_curve=master,
        in_window=copy(a.eval_mask),
    )
end

function cost_function(a::Autoscaler, params)
    if !isnothing(a.eval_window)
        return fixed_window_cost(a, params)
    elseif a.classic_cost_function
        return standard_cost_function(a, params)
    else
        return new_cost_function(a, params)
    end
end
