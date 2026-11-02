function deviation_at(a::Autoscaler, x, selected_size, selected_curve)
    m = better_master_curve(a, x, selected_size)
    ismissing(m) && return missing

    s = interpolate(selected_curve..., x; sq_errs=true)
    ismissing(s) && return missing

    y, dy2 = s
    Y, dY2 = m

    δ = (y - Y)^2 / (dy2 + dY2)
    return δ
end

function master_curve(points, skip)
    numerator = 0.0
    denominator = 0.0

    for (i, (y, dy2)) in enumerate(points)
        i == skip && continue
        weight = 1 / dy2
        numerator += y * weight
        denominator += weight
    end
    return numerator / denominator, 1 / denominator
end

function new_master_curve(a::Autoscaler, x, sz; skip=a.use_all_sizes)
    points = map(a.avail_sizes) do size
        xs, ys, y_errs = scaled_data(a, size)

        interpolate(xs, ys, y_errs, x; sq_errs=true)
    end
    if any(ismissing, points)
        return (NaN, NaN)
    else
        return master_curve(points, skip ? 0 : sz)
    end
end

function new_cost_function_old(a::Autoscaler, params)
    transform!(a, params)
    lo, hi = window(a)

    if lo > hi
        return 1e12 * (lo - hi)
    end
    if hi - lo == 0.0
        return Inf
    end

    skip = !a.use_all_sizes

    result = 0.0
    curr_points = InplacePointIterator(a, lo, hi)
    first_x = curr_points[1].x
    keep_going = next!(curr_points) # for integration, we skip the first point

    last_x = curr_points[1].x
    while keep_going
        prev_points = previous(curr_points)

        curr_x = curr_points[1].x
        prev_x = prev_points[1].x

        for i in eachindex(curr_points)
            curr_δ2 = delta_squared(curr_points, i; skip)
            prev_δ2 = delta_squared(prev_points, i; skip)
            result += (curr_x - prev_x) * (curr_δ2 + prev_δ2) ./ 2
        end

        last_x = curr_points[1].x
        keep_going = next!(curr_points)
    end
    return result / length(curr_points) / (last_x - first_x)
end
function new_cost_function(a::Autoscaler, params)
    transform!(a, params)
    lo, hi = window(a)
    lo > hi && return 1e12 * (lo - hi)
    hi == lo && return Inf

    skip = !a.use_all_sizes

    knots = all_knots(a, lo, hi)
    result = 0.0

    prev_pts = [(0.0, 0.0) for _ in eachindex(a.avail_sizes)]
    curr_pts = [(0.0, 0.0) for _ in eachindex(a.avail_sizes)]

    for (prev_x, curr_x) in zip(knots, Iterators.drop(knots, 1))
        points_at!(prev_pts, a, prev_x)
        points_at!(curr_pts, a, curr_x)
        for i in eachindex(curr_pts)
            curr_δ2 = delta_squared(curr_pts, i; skip)
            prev_δ2 = delta_squared(prev_pts, i; skip)
            result += (curr_x - prev_x) * (curr_δ2 + prev_δ2) / 2
        end
    end
    return result / length(a.avail_sizes) / (knots[end] - knots[1])
end
