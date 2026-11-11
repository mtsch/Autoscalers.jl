# Integral cost function (default): squared deviations from the master curve are integrated
# with the trapezoidal rule over the (shrunk) overlap window of the scaled data.

"""
    all_knots(a::Autoscaler, lo, hi)

All scaled `x` values within `[lo, hi]`, plus the endpoints, sorted and unique.
"""
function all_knots(a::Autoscaler, lo, hi)
    knots = filter(x -> lo ≤ x ≤ hi, a.x_val_scaled)
    append!(knots, (lo, hi))
    unique!(knots)
    sort!(knots)
    return knots
end

"""
    points_at!(dst, a::Autoscaler, x)

Fill `dst` with `(y, dy^2)` of each curve interpolated at scaled `x`.
"""
function points_at!(dst, a::Autoscaler, x)
    length(dst) == n_curves(a) || throw(DimensionMismatch("bad length of `dst`"))
    for i in 1:n_curves(a)
        xs, ys, y_errs = scaled_data(a, i)
        dst[i] = interpolate(xs, ys, y_errs, x; sq_errs=true)
    end
    return dst
end

"""
    delta_squared(points, index; skip=false)

Squared deviation of `points[index]` from the master curve, divided by its variance. If
`skip`, the point itself is left out of the master curve.
"""
function delta_squared(points, index; skip=false)
    y, dy2 = points[index]
    Y, _ = master_curve(points, skip ? index : 0)
    return (y - Y)^2 / dy2
end

"""
    new_master_curve(a::Autoscaler, x, own_curve)

Master curve `(Y, dY2)` at scaled `x`, leaving out `own_curve` unless `a.use_all_sizes`.
Returns `(NaN, NaN)` if any curve does not cover `x`.
"""
function new_master_curve(a::Autoscaler, x, own_curve)
    points = map(1:n_curves(a)) do c
        xs, ys, y_errs = scaled_data(a, c)
        interpolate(xs, ys, y_errs, x; sq_errs=true)
    end
    if any(ismissing, points)
        return (NaN, NaN)
    else
        skip = a.use_all_sizes ? 0 : own_curve
        return master_curve(points, skip)
    end
end

function new_cost_function(a::Autoscaler, params)
    transform!(a, params)
    lo, hi = window(a)
    lo > hi && return 1e12 * (lo - hi)
    hi == lo && return Inf

    skip = !a.use_all_sizes

    knots = all_knots(a, lo, hi)
    result = 0.0

    prev_pts = Vector{Tuple{Float64,Float64}}(undef, n_curves(a))
    curr_pts = Vector{Tuple{Float64,Float64}}(undef, n_curves(a))

    for (prev_x, curr_x) in zip(knots, Iterators.drop(knots, 1))
        points_at!(prev_pts, a, prev_x)
        points_at!(curr_pts, a, curr_x)
        for i in eachindex(curr_pts)
            curr_δ2 = delta_squared(curr_pts, i; skip)
            prev_δ2 = delta_squared(prev_pts, i; skip)
            result += (curr_x - prev_x) * (curr_δ2 + prev_δ2) / 2
        end
    end
    return result / n_curves(a) / (knots[end] - knots[1])
end
