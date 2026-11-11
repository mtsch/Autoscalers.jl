"""
    lerp((x1, y1), (x2, y2), x)

Linearly interpolate (or extrapolate) at `x` on the line through `(x1, y1)` and `(x2, y2)`.

```jldoctest
julia> Autoscalers.lerp((0.0, 1.0), (2.0, 3.0), 1.5)
2.5
```
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

Linearly interpolate `y` and `y_err` between the knots `x` at `selected_x`. With
`sq_errs=true`, the squared errors are interpolated and returned instead. Returns
`(y, y_err)`, or `missing` outside the range of `x`.

```jldoctest
julia> x, y, y_err = [0.0, 1.0, 2.0], [0.0, 10.0, 20.0], [1.0, 1.0, 3.0];

julia> Autoscalers.interpolate(x, y, y_err, 1.5)
(15.0, 2.0)

julia> Autoscalers.interpolate(x, y, y_err, 1.5; sq_errs=true)  # mean of 1² and 3²
(15.0, 5.0)

julia> Autoscalers.interpolate(x, y, y_err, 3.0)
missing
```
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

Like `interpolate(xs, ys, dys, x; sq_errs=true)`, but defined everywhere. Returns
`(y, dy2, d)`. Outside the data:

* `y` continues linearly from the end segment;
* `dy2` is held at its value at the end;
* `d` is the distance outside the data range, in units of the range width (`0` inside).

```jldoctest
julia> xs, ys, dys = [0.0, 1.0, 2.0], [0.0, 10.0, 20.0], [1.0, 1.0, 3.0];

julia> Autoscalers.extrapolate(xs, ys, dys, 1.5)
(15.0, 5.0, 0.0)

julia> Autoscalers.extrapolate(xs, ys, dys, 3.0)  # half a range width beyond the end
(30.0, 9.0, 0.5)
```
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

Error-weighted mean `(Y, dY2)` of `points`, a collection of `(y, dy2)`, leaving out the
point with index `skip` (`skip = 0` uses all).

```jldoctest
julia> points = [(1.0, 1.0), (3.0, 1.0), (100.0, 4.0)];

julia> Autoscalers.master_curve(points, 3)
(2.0, 0.5)

julia> Autoscalers.master_curve(points, 0)  # the less precise third point pulls less
(12.88888888888889, 0.4444444444444444)
```
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
