"""
    Autoscaler(
        table, ansatz::ScalingAnsatz=DEFAULT_ANSATZ;
        x=:x, y=:y, y_err=:y_err, L=:L,
        classic_cost_function=false,
        use_all_sizes=false,
        window_width=1,
        eval_window=nothing,
        penalty=100.0,
        params...
    )

Prepare `table` for finite-size scaling with `ansatz`. Calling the result with the free
parameters (in the order of `ansatz.param_names`, skipping hardcoded ones) returns the cost.

* `x`, `y`, `y_err`, `L`: column names. Defaults are taken from the remaps of `ansatz`.
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
* `params...`: hardcode parameters by name, e.g. `a = 1/ν`.

Rows with missing or non-finite values, or with `y_err ≤ 0`, are skipped with a warning.
System sizes must be positive integers, each size needs at least two usable rows, and
no two rows may share the same `L` and `x`.

The `Autoscaler` reuses internal buffers between calls, so a single instance must not be
evaluated from several threads at once.

# References

- J. Houdayer and A. K. Hartmann, *Low-temperature behavior of two-dimensional Gaussian
  Ising spin glasses*, Phys. Rev. B **70**, 014418 (2004),
  [doi:10.1103/PhysRevB.70.014418](https://doi.org/10.1103/PhysRevB.70.014418).
"""
struct Autoscaler{P,S<:ScalingAnsatz{P}}
    ansatz::S

    x_val_raw::Vector{Float64}
    y_val_raw::Vector{Float64}
    y_err_raw::Vector{Float64}

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
    table, ansatz::ScalingAnsatz=DEFAULT_ANSATZ;
    x=nothing, y=nothing, y_err=nothing, L=nothing,
    classic_cost_function=false,
    use_all_sizes=false,
    window_width=1,
    eval_window=nothing,
    penalty=100.0,
    kwargs...
)
    x = something(x, get(ansatz.remaps, :x, :x))
    y = something(y, get(ansatz.remaps, :y, :y))
    y_err = something(y_err, get(ansatz.remaps, :y_err, :y_err))
    L = something(L, get(ansatz.remaps, :L, :L))

    for (k, _) in kwargs
        if k ∉ ansatz.param_names
            throw(ArgumentError("unrecognized keyword argument `$k`"))
        end
    end
    if !isnothing(eval_window) && classic_cost_function
        throw(ArgumentError("`eval_window` and `classic_cost_function` are mutually exclusive"))
    end
    0 < window_width ≤ 1 || throw(ArgumentError("`window_width` must be in (0, 1]"))

    n_missing = n_nonfinite = n_bad_err = 0
    rows = filter(collect(Tables.rows(table))) do row
        values = (row[L], row[x], row[y], row[y_err])
        if any(ismissing, values)
            n_missing += 1
        elseif !all(isfinite, values)
            n_nonfinite += 1
        elseif row[y_err] ≤ 0
            n_bad_err += 1
        else
            l = row[L]
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
    n_bad_err > 0 && @warn "Skipping $n_bad_err rows with non-positive `$y_err`."
    sort!(rows; by=r -> (r[L], r[x]))

    x_val_raw = Float64[]
    y_val_raw = Float64[]
    y_err_raw = Float64[]
    sizes = Int[]
    avail_sizes = Int[]
    size_map = Dict{Int,UnitRange{Int}}()

    prev_size = -1
    start_index = 0
    for (i, row) in enumerate(rows)
        curr_size = Int(row[L])
        if curr_size ≠ prev_size
            if prev_size > 0
                size_map[prev_size] = start_index:(i - 1)
            end
            push!(avail_sizes, curr_size)
            prev_size = curr_size
            start_index = i
        elseif row[x] == last(x_val_raw)
            # Interpolation and the classic cost need strictly increasing x within a size.
            throw(ArgumentError("duplicate rows for L=$curr_size, `$x`=$(row[x])"))
        end
        push!(x_val_raw, row[x])
        push!(y_val_raw, row[y])
        push!(y_err_raw, row[y_err])
        push!(sizes, curr_size)
    end
    isempty(avail_sizes) && throw(ArgumentError("no usable data"))
    length(avail_sizes) ≥ 2 || throw(ArgumentError("need at least two system sizes"))
    size_map[prev_size] = start_index:length(x_val_raw)
    for l in avail_sizes
        if length(size_map[l]) < 2
            throw(ArgumentError("system size L=$l has fewer than two usable data points"))
        end
    end

    # The mask is built here, after sorting and dropping unusable rows, so its indices
    # always match the internal arrays.
    if isnothing(eval_window)
        ew = nothing
        eval_mask = trues(length(x_val_raw))
    else
        ew = (Float64(eval_window[1]), Float64(eval_window[2]))
        ew[1] < ew[2] || throw(ArgumentError("`eval_window` must satisfy lo < hi"))
        eval_mask = BitVector(ew[1] .≤ x_val_raw .≤ ew[2])
        for l in avail_sizes
            if !any(view(eval_mask, size_map[l]))
                @warn "System size L=$l has no data points inside `eval_window`; " *
                      "it will only be used for interpolation."
            end
        end
        any(eval_mask) || throw(ArgumentError("no data points inside `eval_window`"))
    end

    result = Autoscaler(
        ansatz,
        x_val_raw, y_val_raw, y_err_raw,
        copy(x_val_raw), copy(y_val_raw), copy(y_err_raw),
        sizes, avail_sizes, size_map,
        Float64[], Float64[], Float64[],
        fill(NaN, length(ansatz.param_names)), zeros(length(ansatz.param_names)),
        classic_cost_function, use_all_sizes, Float64(window_width),
        ew, eval_mask, Float64(penalty),
    )
    return hardcode_params!(result; kwargs...)
end

function hardcode_params!(a::Autoscaler; kwargs...)
    for (i, p) in enumerate(a.ansatz.param_names)
        a.hardcodes[i] = get(kwargs, p, NaN)
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

function transform!(a::Autoscaler, params)
    for i in eachindex(a.x_val_scaled)
        L, x, y, y_err = a.sizes[i], a.x_val_raw[i], a.y_val_raw[i], a.y_err_raw[i]

        a.x_val_scaled[i] = a.ansatz.scale_x(x, L, params)

        y_scaled = a.ansatz.scale_y(x, y ± y_err, L, params)
        a.y_val_scaled[i] = Measurements.value(y_scaled)
        a.y_err_scaled[i] = Measurements.uncertainty(y_scaled)
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
