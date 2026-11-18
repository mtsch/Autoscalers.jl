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
    interpolate(x, y, y_err, selected_x; exact=false)
    interpolate(x, y, selected_x)

Linearly interpolate `y` between the knots `x` at `selected_x`, and return `(y, variance)`,
or `missing` outside the range of `x`. The variance interpolates `y_err²` linearly or,
with `exact=true`, is the variance of the interpolant of independent points,
``(1 - λ)^2 σ_k^2 + λ^2 σ_{k+1}^2`` with ``λ = (x_\\mathrm{sel} - x_k)/(x_{k+1} - x_k)``.
Without `y_err`, only the interpolated `y` is returned.

```jldoctest
julia> x, y, y_err = [0.0, 1.0, 2.0], [0.0, 10.0, 20.0], [1.0, 1.0, 3.0];

julia> Autoscalers.interpolate(x, y, y_err, 1.5)  # mean of 1² and 3²
(15.0, 5.0)

julia> Autoscalers.interpolate(x, y, y_err, 1.5; exact=true)  # (1² + 3²) / 4
(15.0, 2.5)

julia> Autoscalers.interpolate(x, y, 1.5)
15.0

julia> Autoscalers.interpolate(x, y, y_err, 3.0)
missing
```
"""
# Index `k` of the segment `x[k] ≤ selected_x < x[k + 1]`, or of the last knot if
# `selected_x` is on it; 0 outside the data.
function segment(x, selected_x)
    k = searchsortedlast(x, selected_x)
    return k == length(x) && x[k] ≠ selected_x ? 0 : k
end

function interpolate(x, y, y_err, selected_x; exact=false)
    k = segment(x, selected_x)
    k == 0 && return missing
    x[k] == selected_x && return y[k], y_err[k]^2
    value = lerp((x[k], y[k]), (x[k + 1], y[k + 1]), selected_x)
    if exact
        λ = (selected_x - x[k]) / (x[k + 1] - x[k])
        return value, (1 - λ)^2 * y_err[k]^2 + λ^2 * y_err[k + 1]^2
    else
        return value, lerp((x[k], y_err[k]^2), (x[k + 1], y_err[k + 1]^2), selected_x)
    end
end
function interpolate(x, y, selected_x)
    k = segment(x, selected_x)
    k == 0 && return missing
    x[k] == selected_x && return y[k]
    return lerp((x[k], y[k]), (x[k + 1], y[k + 1]), selected_x)
end

"""
    extrapolate(xs, ys, dys, x; exact=false)

Like `interpolate(xs, ys, dys, x; exact)`, but defined everywhere. Outside the data, `y`
continues linearly from the end segment and the variance is held at its value at the end.

```jldoctest
julia> xs, ys, dys = [0.0, 1.0, 2.0], [0.0, 10.0, 20.0], [1.0, 1.0, 3.0];

julia> Autoscalers.extrapolate(xs, ys, dys, 1.5)
(15.0, 5.0)

julia> Autoscalers.extrapolate(xs, ys, dys, 3.0)
(30.0, 9.0)
```
"""
function extrapolate(xs, ys, dys, x; exact=false)
    n = length(xs)
    if x < xs[1]
        return lerp((xs[1], ys[1]), (xs[2], ys[2]), x), dys[1]^2
    elseif x > xs[end]
        return lerp((xs[n - 1], ys[n - 1]), (xs[n], ys[n]), x), dys[n]^2
    else
        y, dy2 = interpolate(xs, ys, dys, x; exact)
        return y, dy2
    end
end
