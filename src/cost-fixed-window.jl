# Fixed-window cost function: a fixed set of points, chosen in raw `x`, is compared to the
# master curve of the other curves. Where another curve does not reach a point, it is
# extrapolated linearly; `fit_collapse` warns when the fitted collapse relies on that.

"""
    fixed_master_curve(a::Autoscaler, x, own_curve; exact=false)

Error-weighted master curve at scaled `x`, built from all curves except `own_curve`
(unless `a.use_all_sizes`), extrapolating where needed. Returns `(; Y, dY2)`. `exact`
selects the variance of each curve's interpolant, as in [`extrapolate`](@ref).
"""
function fixed_master_curve(a::Autoscaler, x, own_curve; exact=false)
    num = den = 0.0
    for c in 1:n_curves(a)
        (!a.use_all_sizes && c == own_curve) && continue
        xs, ys, dys = scaled_data(a, c)
        y, dy2 = extrapolate(xs, ys, dys, x; exact)
        num += y / dy2
        den += 1 / dy2
    end
    return (; Y=num / den, dY2=1 / den)
end

"""
    fixed_window_cost(a::Autoscaler, params)

Fixed-window cost: for each curve, the mean ``χ^2`` of its scored points against the
master curve of the other curves, averaged over the curves with at least one scored point.
"""
function fixed_window_cost(a::Autoscaler, params)
    transform!(a, params)
    total = 0.0
    n_scored = 0
    for (c, range) in enumerate(a.curve_ranges)
        chi2 = 0.0
        n = 0
        for i in range
            a.eval_mask[i] || continue
            m = fixed_master_curve(a, a.x_val_scaled[i], c)
            chi2 += (a.y_val_scaled[i] - m.Y)^2 / (a.y_err_scaled[i]^2 + m.dY2)
            n += 1
        end
        n == 0 && continue
        total += chi2 / n
        n_scored += 1
    end
    return total / n_scored
end

# Whether scaled `x` lies within the data of every curve other than `own_curve`.
function covered(a::Autoscaler, x, own_curve)
    return all(1:n_curves(a)) do c
        xs = scaled_data(a, c).x_val
        return c == own_curve || xs[1] ≤ x ≤ xs[end]
    end
end

# At the current scaling, the number of scored points (sum) or integrated curves (integral)
# at which the window cost extrapolates another curve beyond its data.
function n_extrapolated(a::Autoscaler)
    if a.cost == :sum
        return count(eachindex(a.x_val_scaled)) do i
            return a.eval_mask[i] && !covered(a, a.x_val_scaled[i], a.curve[i])
        end
    else
        # The data of each curve span an interval, so covering both ends of an
        # integration range covers all of it.
        return count(1:n_curves(a)) do c
            r = integration_range(a, c)
            return !isnothing(r) && !(covered(a, r[1], c) && covered(a, r[2], c))
        end
    end
end

"""
    extrapolates(a::Autoscaler, args)

Whether the window cost at the free parameters `args` extrapolates curves beyond their
data: for the sum, at a scored point; for the integral, anywhere in a curve's integration
range. Always `false` for the classic cost.
"""
function extrapolates(a::Autoscaler, args)
    a.cost == :classic && return false
    transform!(a, get_parameters!(a, args))
    return n_extrapolated(a) > 0
end

# Unscaled x of a curve with data `(ts, xs)` at scaled `x`, moved by whole floats with
# `step` (`nextfloat` or `prevfloat`) until mapping it back does not cross `x`: the two
# interpolations do not invert each other exactly in floating point.
function unscaled_at(xs, ts, x, step)
    t = interpolate(xs, ts, x)::Float64
    while (step === nextfloat ? (<) : (>))(interpolate(ts, xs, t)::Float64, x)
        t = step(t)
    end
    return t
end

"""
    extrapolation_free_window(a::Autoscaler, args)

At the free parameters `args`, the largest part `(lo, hi)` of `eval_window` in which the
window cost needs no extrapolation, or `nothing` if there is none.

Each curve that the window cost uses is covered by all other curves for scaled `x`
between the largest first and the smallest last scaled `x` of the others. That range is
mapped back to the curve's unscaled `x`, and these intervals are intersected.
"""
function extrapolation_free_window(a::Autoscaler, args)
    transform!(a, get_parameters!(a, args))
    ts_all = raw_x(a)
    lo = max(a.eval_window[1], minimum(ts_all))
    hi = min(a.eval_window[2], maximum(ts_all))
    for c in 1:n_curves(a)
        uses_window(a, c) || continue
        others = [scaled_data(a, o).x_val for o in 1:n_curves(a) if o ≠ c]
        x_lo, x_hi = maximum(first, others), minimum(last, others)
        range = a.curve_ranges[c]
        xs, ts = view(a.x_val_scaled, range), view(ts_all, range)
        # Nowhere covered: the window would have to avoid this curve's data altogether.
        (x_lo > x_hi || x_lo > xs[end] || x_hi < xs[1]) && return nothing
        if x_lo > xs[1]
            lo = max(lo, unscaled_at(xs, ts, x_lo, nextfloat))
        end
        if x_hi < xs[end]
            hi = min(hi, unscaled_at(xs, ts, x_hi, prevfloat))
        end
    end
    hi > lo || return nothing
    # The window cost must still have something to score in the suggested window.
    in_window(range) = uses_window(
        a.cost, view(ts_all, range), lo .≤ view(ts_all, range) .≤ hi, (lo, hi)
    )
    return any(in_window, a.curve_ranges) ? (lo, hi) : nothing
end

function warn_extrapolation(a::Autoscaler, args)
    a.cost == :classic && return nothing
    transform!(a, get_parameters!(a, args))
    n = n_extrapolated(a)
    n == 0 && return nothing
    if a.cost == :sum
        location = "at $n of $(count(a.eval_mask)) scored points"
    else
        n_used = count(c -> uses_window(a, c), 1:n_curves(a))
        location = "in the integration ranges of $n of $n_used curves"
    end
    window = extrapolation_free_window(a, args)
    if isnothing(window)
        suggestion = "no part of `eval_window` is free of it"
    else
        suggestion = "`eval_window=$window` would need none"
    end
    @warn "The fitted collapse extrapolates curves beyond their data $location, so its " *
          "cost compares them with extrapolated master curves; $suggestion."
    return nothing
end
