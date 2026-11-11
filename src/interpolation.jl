"""
    lerp((x1, y1), (x2, y2), x)

Linearly interpolate (or extrapolate) at `x` on the line through `(x1, y1)` and `(x2, y2)`.
"""
function lerp((x1, y1), (x2, y2), selected_x)
    if x1 == x2
        return y1
    else
        return (y1 * (x2 - selected_x) + y2 * (selected_x - x1)) / (x2 - x1)
    end
end

"""
    interpolate(x, y, y_err, selected_x; sq_errs=false)

Interpolate `y` and `y_err` (squared before interpolating if `sq_errs=true`) with knots
in `x` at point `selected_x`. Returns `missing` outside the range of `x`.
"""
function interpolate(x, y, y_err, selected_x; sq_errs=false)
    indices = searchsorted(x, selected_x)
    if indices.start ≤ indices.stop
        # exact hit on a knot
        e = y_err[indices.start]
        return y[indices.start], sq_errs ? e^2 : e
    elseif indices.start > length(x) || indices.stop ≤ 0
        return missing
    else
        # `indices` is an empty range: stop < start
        left = indices.stop
        right = indices.start
        e1, e2 = y_err[left], y_err[right]
        if sq_errs
            e1 *= e1
            e2 *= e2
        end
        return (
            lerp((x[left], y[left]), (x[right], y[right]), selected_x),
            lerp((x[left], e1), (x[right], e2), selected_x),
        )
    end
end

"""
    extrapolate(xs, ys, dys, x)

Like `interpolate(xs, ys, dys, x; sq_errs=true)`, but defined everywhere. Outside the
data, `y` is extended linearly from the end segment and `dy^2` is held at its end value
(both continuous at the edge). Returns `(y, dy2, d)`, where `d` is the distance outside
the data range in units of the range width (`0` inside).
"""
function extrapolate(xs, ys, dys, x)
    n = length(xs)
    n ≥ 2 || throw(ArgumentError("need at least two points per curve"))
    width = xs[end] - xs[1]
    if x < xs[1]
        y = lerp((xs[1], ys[1]), (xs[2], ys[2]), x)
        return y, dys[1]^2, (xs[1] - x) / width
    elseif x > xs[end]
        y = lerp((xs[n-1], ys[n-1]), (xs[n], ys[n]), x)
        return y, dys[n]^2, (x - xs[end]) / width
    else
        y, dy2 = interpolate(xs, ys, dys, x; sq_errs=true)
        return y, dy2, 0.0
    end
end

"""
    master_curve(points, skip)

Error-weighted mean of `points`, a collection of `(y, dy2)`, leaving out index `skip`
(`skip = 0` uses all). Returns `(Y, dY2)`.
"""
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
