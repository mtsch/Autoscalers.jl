"""
    Autoscaler(
        table, ansatz::Ansatz=DEFAULT_ANSATZ;
        eval_window=nothing,
        classic_cost_function=false,
        use_all_sizes=false,
        fix=(;),
        integral=:none,
    )

Prepare the data in `table` (any Tables.jl table) for a data collapse with `ansatz`. The
default ansatz is `@ansatz (a, b, x_c) -> :L^b * (:y ± :y_err) ~ :L^a * (:x - x_c)`.

Calling the `Autoscaler` with values of the free parameters returns the cost of the
collapse, which [`fit_collapse`](@ref) minimises. The parameters are those of
`ansatz.param_names`, in that order, without the fixed ones.

# Cost functions

Each data point is compared with the master curve built from all *other* curves (a curve
is one size, or one combination of size and `ansatz.group` values). One of the two cost
functions must be chosen:

* **Fixed window** (`eval_window = (lo, hi)`): only points with
  ``\\mathrm{lo} ≤ x ≤ \\mathrm{hi}`` in the *unscaled* ``x`` are scored, so the scored set
  does not change with the parameters. Points outside it still serve for interpolation.
  Where another curve does not reach a point's scaled ``x``, it is extrapolated linearly,
  with the variance held at its value at the end. Nothing is added to the cost for this,
  so a window wider than the range the curves have in common can bias the fit;
  [`fit_collapse`](@ref) warns when the fitted collapse extrapolates and suggests a
  window that needs no extrapolation.
* **Classic** (`classic_cost_function=true`): the cost of Houdayer & Hartmann (2004), with
  a local linear fit through the neighbouring points of the other curves. It scores every
  point where the curves overlap in the *scaled* ``x``, so for the smallest sizes it
  reaches furthest from the transition, where corrections to scaling are largest.

The scaling form holds as ``t → 0`` in the unscaled ``t``, for any scaled ``x``. A
window in the unscaled ``x`` bounds the corrections that grow with ``|t|`` equally for all
sizes, so the fixed window is recommended.

# Keywords

* `use_all_sizes`: also include a point's own curve in its master curve (fixed-window cost
  only).
* `fix`: parameters held at fixed values, e.g. `fix=(; a=1/ν)`. They are left out of the
  arguments of the cost.
* `integral` (experimental, fixed-window cost only): instead of summing over the points
  in the window, integrate the ``χ^2`` density of each curve's linear interpolant against
  the master curve over the part of the window the curve covers. The integral is taken
  in the scaled ``x`` and divided by the scaled length of that part; for an ansatz with
  ``x`` affine in the unscaled ``x``, such as ``L^a (x - x_c)``, this is the mean over
  the window. It is evaluated with adaptive Gauss–Kronrod quadrature (QuadGK.jl)
  between the points of all curves. The sum has a kink in the parameters wherever a point
  of one curve crosses a point of another in scaled ``x``; the integral does not. `:gk`
  interpolates ``σ^2`` linearly, so a correct collapse costs about ``2/3``; `:gk_exact`
  uses the variance of the interpolant, ``(1 - λ)^2 σ_k^2 + λ^2 σ_{k+1}^2`` with
  ``λ = (x - x_k)/(x_{k+1} - x_k)``, so it costs about 1. The default `:none` is the sum.
  A curve takes part if its data cover a part of the window of positive length, even
  without a data point inside it.

# Data requirements

* Only the columns used by `ansatz` are read. Rows with missing or non-finite values in
  them, or with a non-positive error, are skipped with a warning.
* Sizes must be positive integers, and there must be at least two of them.
* Each curve needs at least two rows, with distinct `x`.

An `Autoscaler` reuses internal buffers, so one instance must not be evaluated from several
threads at once; see `Autoscaler(::Autoscaler)` for independent copies.

# Examples

```jldoctest
julia> data = [(; L, x, y=L^-0.25 * tanh(L * (x - 0.3)), y_err=1e-3)
               for L in (8, 16, 32) for x in 0.2:0.005:0.4];

julia> s = Autoscaler(data; eval_window=(0.28, 0.32));

julia> round(s(1.0, 0.25, 0.3); digits=3)  # a, b, x_c: a good collapse
0.005

julia> round(s(1.2, 0.25, 0.3); digits=1)  # a bad one
270.2

julia> s_b = Autoscaler(data; eval_window=(0.28, 0.32), fix=(; b=0.25));

julia> s_b(1.0, 0.3) == s(1.0, 0.25, 0.3)
true
```

# References

- J. Houdayer and A. K. Hartmann, *Low-temperature behavior of two-dimensional Gaussian
  Ising spin glasses*, Phys. Rev. B **70**, 014418 (2004),
  [doi:10.1103/PhysRevB.70.014418](https://doi.org/10.1103/PhysRevB.70.014418).
"""
struct Autoscaler{P,A<:Ansatz{P},D<:NamedTuple,E<:NamedTuple}
    ansatz::A

    # Columns used by the ansatz, sorted by curve and `x`, and for each of them `nothing` or
    # the index of its partial derivative (see `error_dual`) and its errors.
    data::D
    errors::E

    x_val_scaled::Vector{Float64}
    y_val_scaled::Vector{Float64}
    y_err_scaled::Vector{Float64}
    sizes::Vector{Int}

    # Curve index of each row, the rows of each curve, and a name for messages.
    curve::Vector{Int}
    curve_ranges::Vector{UnitRange{Int}}
    curve_labels::Vector{String}

    # Values of the fixed parameters, NaN for the free ones; the buffer for the full
    # parameter vector.
    fixed::Vector{Float64}
    last_params::Vector{Float64}

    # :classic, :sum (fixed window), or the windowed integrals :gk and :gk_exact.
    cost::Symbol
    use_all_sizes::Bool

    eval_window::Union{Nothing,Tuple{Float64,Float64}}
    eval_mask::BitVector
end

function Autoscaler(
    table, ansatz::Ansatz=DEFAULT_ANSATZ;
    classic_cost_function=false,
    use_all_sizes=false,
    eval_window=nothing,
    fix=(;),
    integral=:none,
)
    check_fix(ansatz, fix)
    if classic_cost_function
        if !isnothing(eval_window)
            throw(ArgumentError(
                "`eval_window` and `classic_cost_function` are mutually exclusive"
            ))
        end
        if use_all_sizes
            throw(ArgumentError("`use_all_sizes` applies to the fixed-window cost only"))
        end
        if integral ≠ :none
            throw(ArgumentError("`integral` applies to the fixed-window cost only"))
        end
        cost = :classic
    elseif isnothing(eval_window)
        throw(ArgumentError(
            "choose a cost function: `eval_window=(lo, hi)` scores the points with " *
            "lo ≤ x ≤ hi, or `classic_cost_function=true`"
        ))
    elseif integral == :none
        cost = :sum
    elseif integral in (:gk, :gk_exact)
        cost = integral
    else
        throw(ArgumentError("`integral` must be :none, :gk or :gk_exact"))
    end

    L = ansatz.size
    x = ansatz.x
    columns = ansatz.columns
    error_columns = Tuple(ansatz.errors)
    used = (columns..., error_columns...)
    available = Tables.columnnames(Tables.columns(table))
    absent = [c for c in used if c ∉ available]
    if !isempty(absent)
        throw(ArgumentError(
            "the table has no column " * join(("`$c`" for c in absent), ", ") *
            " used by the ansatz"
        ))
    end
    n_missing = n_nonfinite = n_bad_err = 0
    rows = filter(collect(Tables.rows(table))) do row
        values = map(c -> Tables.getcolumn(row, c), used)
        if any(ismissing, values)
            n_missing += 1
        elseif any(v -> !(v isa Real), values)
            k = findfirst(v -> !(v isa Real), values)
            throw(ArgumentError(
                "column `$(used[k])` must hold real numbers, got $(repr(values[k]))"
            ))
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
    group = ansatz.group
    # `+ 0.0` turns `-0.0` into `0.0`: sorting tells them apart but `==` does not, so
    # they would otherwise end up as one curve with its rows out of order.
    function curve_key(r)
        return (Tables.getcolumn(r, L), map(c -> Tables.getcolumn(r, c) + 0.0, group)...)
    end
    sort!(rows; by=r -> (curve_key(r)..., Tables.getcolumn(r, x)))

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

    isempty(rows) && throw(ArgumentError("no usable data"))
    curve = Vector{Int}(undef, length(rows))
    curve_ranges = UnitRange{Int}[]
    curve_labels = String[]
    for (i, r) in enumerate(rows)
        if i == 1 || curve_key(r) ≠ curve_key(rows[i - 1])
            push!(curve_ranges, i:i)
            label = join(("$n=$v" for (n, v) in zip((L, group...), curve_key(r))), ", ")
            push!(curve_labels, label)
        elseif x_raw[i] == x_raw[i - 1]
            # Interpolation and the classic cost need strictly increasing x within a curve.
            label = curve_labels[end]
            throw(ArgumentError("duplicate rows for $label, `$x`=$(x_raw[i])"))
        else
            curve_ranges[end] = first(curve_ranges[end]):i
        end
        curve[i] = length(curve_ranges)
    end
    length(unique(sizes)) ≥ 2 || throw(ArgumentError("need at least two system sizes"))
    for (range, label) in zip(curve_ranges, curve_labels)
        if length(range) < 2
            throw(ArgumentError("$label has fewer than two usable data points"))
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
        uses(r) = uses_window(cost, view(x_raw, r), ew)
        what = cost == :sum ? "data points inside" : "data covering a part of"
        for (range, label) in zip(curve_ranges, curve_labels)
            if !uses(range)
                @warn "$label has no $what `eval_window`; it will only be used for " *
                      "interpolation."
            end
        end
        any(uses, curve_ranges) || throw(ArgumentError("no $what `eval_window`"))
    end

    n = length(sizes)
    n_params = length(ansatz.param_names)
    result = Autoscaler(
        ansatz, data, errors,
        zeros(n), zeros(n), zeros(n),
        sizes, curve, curve_ranges, curve_labels,
        fill(NaN, n_params), zeros(n_params),
        cost, use_all_sizes, ew, eval_mask,
    )
    return fix_params!(result, fix)
end

# Whether the window cost scores curve data with raw `x` values `ts`: the sum needs a point
# in `window`, the integral a part of it of positive length.
function uses_window(cost, ts, (lo, hi))
    if cost == :sum
        return any(t -> lo ≤ t ≤ hi, ts)
    else
        return !isnothing(raw_overlap(ts, (lo, hi)))
    end
end
function uses_window(a::Autoscaler, c)
    return uses_window(a.cost, view(raw_x(a), a.curve_ranges[c]), a.eval_window)
end

# NaN marks a free parameter in `fixed`, so it cannot be a fixed value.
function check_fix(ansatz::Ansatz, fix)
    if !(fix isa NamedTuple)
        throw(ArgumentError("`fix` must be a `NamedTuple`, e.g. `(; a=1.5)`"))
    end
    for (k, v) in pairs(fix)
        k in ansatz.param_names || throw(ArgumentError("`$k` in `fix` is not a parameter"))
        isnan(v) && throw(ArgumentError("`$k` in `fix` must not be NaN"))
    end
    return nothing
end

function fix_params!(a::Autoscaler, fix::NamedTuple)
    for (name, value) in pairs(fix)
        a.fixed[findfirst(==(name), a.ansatz.param_names)] = value
    end
    return a
end

"""
    Autoscaler(a::Autoscaler; data=a.data, fix=(;))

A copy of `a` with its own buffers, so it can be evaluated concurrently with `a`.

* `data` replaces the data columns. It must have the same columns and rows as `a.data`,
  in the same order, and only the values in columns other than `x`, the size and the
  `group` columns may differ, since those define the curves and the window. The values
  must be finite.
* `fix` fixes parameters in addition to those already fixed in `a`; for a parameter
  that is fixed already, it replaces the value.

# Examples

```jldoctest
julia> data = [(; L, x, y=L^-0.25 * tanh(L * (x - 0.3)), y_err=1e-3)
               for L in (8, 16, 32) for x in 0.2:0.005:0.4];

julia> s = Autoscaler(data; eval_window=(0.28, 0.32));

julia> c = Autoscaler(s; fix=(; b=0.25));

julia> c(1.0, 0.3) == s(1.0, 0.25, 0.3)
true
```
"""
function Autoscaler(a::Autoscaler; data=a.data, fix=(;))
    check_fix(a.ansatz, fix)
    n = length(a.sizes)
    if keys(data) ≠ keys(a.data) || any(v -> length(v) ≠ n, data)
        throw(ArgumentError("`data` must have the same columns and rows as `a.data`"))
    end
    for c in keys(data)
        if c in (a.ansatz.size, a.ansatz.x, a.ansatz.group...)
            if data[c] ≠ a.data[c]
                throw(ArgumentError("`data` must not change column `$c`"))
            end
        elseif !all(isfinite, data[c])
            throw(ArgumentError("column `$c` of `data` must be finite"))
        end
    end
    # Fresh buffers and the new data; every other field is shared with `a`.
    replaced = (;
        data,
        x_val_scaled=zeros(n), y_val_scaled=zeros(n), y_err_scaled=zeros(n),
        fixed=copy(a.fixed), last_params=zeros(length(a.last_params)),
    )
    fields = (get(replaced, f, getfield(a, f)) for f in fieldnames(Autoscaler))
    return fix_params!(Autoscaler(fields...), fix)
end

Base.show(io::IO, a::Autoscaler) = print(io, "Autoscaler(", a.ansatz, ")")

"""
    free_parameter_names(a::Autoscaler)

Names of the parameters that are not fixed, in the order the cost function expects them.
"""
function free_parameter_names(a::Autoscaler)
    return Tuple(p for (p, v) in zip(a.ansatz.param_names, a.fixed) if isnan(v))
end

# Tags the dual numbers that carry the data errors, as `ForwardDiff.Tag{ErrorPropagation}`,
# so they cannot be confused with duals of other differentiations.
struct ErrorPropagation end
const ErrorTag = ForwardDiff.Tag{ErrorPropagation}

"""
    error_dual(tag, value, (slot, errors), i, ::Val{K})

`value` as a dual number whose `slot`-th of `K` partials is its error. Propagated through
the ansatz, the partials of the scaled ``y`` are then ``σ_c \\, ∂y/∂c`` for each column
``c`` with errors, and their norm is the first-order error of the scaled ``y``.
"""
function error_dual(::T, value, (slot, errors), i, ::Val{K}) where {T,K}
    return Dual{T}(value, ntuple(j -> j == slot ? errors[i] : 0.0, Val(K)))
end

function transform!(a::Autoscaler, params)
    p = parameter_tuple(a.ansatz, params)
    f = a.ansatz.f
    n_errors = Val(length(a.ansatz.errors))
    # ForwardDiff nests duals by the order in which their tags were first created. Creating
    # the tag here, before the ansatz runs, makes it older than any tag the ansatz creates
    # itself, so an ansatz may use ForwardDiff on columns that carry errors. The parameters
    # are always `Float64` (`get_parameters!` writes them into a buffer), so the cost itself
    # cannot be differentiated with ForwardDiff.
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
    # Interpolation and the overlap window assume ascending scaled `x` within each curve.
    for (range, label) in zip(a.curve_ranges, a.curve_labels)
        if !issorted(view(a.x_val_scaled, range))
            throw(ArgumentError(
                "scaled `x` is not ascending for $label at parameters $params; the " *
                "ansatz must map `x` to an increasing function of `x` for every curve"
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

n_curves(a::Autoscaler) = length(a.curve_ranges)

function scaled_data(a::Autoscaler, curve)
    range = a.curve_ranges[curve]
    return (
        x_val=view(a.x_val_scaled, range),
        y_val=view(a.y_val_scaled, range),
        y_err=view(a.y_err_scaled, range),
    )
end

"""
    get_parameters(a::Autoscaler, args)

Combine the free parameters `args` with the fixed ones into the full parameter vector.
"""
get_parameters(a::Autoscaler, args) = copy(get_parameters!(a, args))

"""
    get_parameters!(a::Autoscaler, args)

Like [`get_parameters`](@ref), but returns an internal buffer that is overwritten on the
next call. Avoids an allocation per cost evaluation.
"""
function get_parameters!(a::Autoscaler, args)
    expected = count(isnan, a.fixed)
    if length(args) ≠ expected
        throw(ArgumentError("expected $(expected) params, got $(length(args))"))
    end

    params = a.last_params
    arg_idx = 1
    for i in eachindex(params)
        if isnan(a.fixed[i])
            params[i] = args[arg_idx]
            arg_idx += 1
        else
            params[i] = a.fixed[i]
        end
    end
    return params
end

(a::Autoscaler)(params::AbstractVector) = cost_function(a, get_parameters!(a, params))
(a::Autoscaler)(args::Vararg{Real}) = cost_function(a, get_parameters!(a, args))

"""
    to_table(a::Autoscaler, args...)
    to_table(a::Autoscaler, params::AbstractVector)

The data scaled with the free parameters `args` (or a vector of them), for plotting the
collapse. Returns a `NamedTuple` of columns, one row per data point:

* `L`: system size;
* the `group` columns of the ansatz, if any;
* `x`, `y`, `y_err`: scaled values;
* `master_curve`: the master curve the point is compared to (`NaN` where the classic
  cost has none, because no other curve brackets the point);
* `in_window`: whether the unscaled `x` lies in `eval_window` (always `true` for the
  classic cost). These are the points the fixed-window sum scores; the integral uses
  the whole part of the window each curve covers.

The column names are always these, whatever the columns are called in the data. With
DataFrames.jl loaded, `DataFrame(a, args)` returns the same as a `DataFrame`.

# Examples

```jldoctest
julia> data = [(; L, x, y=L^-0.25 * tanh(L * (x - 0.3)), y_err=1e-3)
               for L in (8, 16, 32) for x in 0.2:0.005:0.4];

julia> t = to_table(Autoscaler(data; eval_window=(0.28, 0.32)), 1.0, 0.25, 0.3);

julia> keys(t)
(:L, :x, :y, :y_err, :master_curve, :in_window)

julia> count(t.in_window)
27
```
"""
to_table(a::Autoscaler, args::Vararg{Real}) = to_table(a, args)
function to_table(a::Autoscaler, args::Union{AbstractVector,Tuple})
    params = get_parameters!(a, args)
    transform!(a, params)
    if a.cost == :classic
        master = map(zip(a.x_val_scaled, a.curve)) do (x, c)
            m = standard_master_curve(a, x, c)
            ismissing(m) ? NaN : m[1]
        end
    else
        exact = a.cost == :gk_exact
        master = map(zip(a.x_val_scaled, a.curve)) do (x, c)
            return fixed_master_curve(a, x, c; exact).Y
        end
    end
    group = a.ansatz.group
    return (;
        L=copy(a.sizes),
        NamedTuple{group}(map(c -> copy(a.data[c]), group))...,
        x=copy(a.x_val_scaled),
        y=copy(a.y_val_scaled),
        y_err=copy(a.y_err_scaled),
        master_curve=master,
        in_window=copy(a.eval_mask),
    )
end

function cost_function(a::Autoscaler, params)
    if a.cost == :classic
        return standard_cost_function(a, params)
    elseif a.cost == :sum
        return fixed_window_cost(a, params)
    else
        return integral_window_cost(a, params)
    end
end
