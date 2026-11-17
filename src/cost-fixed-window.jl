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

"""
    extrapolates(a::Autoscaler, args)

Whether the window cost at the free parameters `args` extrapolates curves beyond their
data: for the sum, at a scored point; for the integral, anywhere in a curve's integration
range. Always `false` for the classic cost.
"""
function extrapolates(a::Autoscaler, args)
    isnothing(a.eval_window) && return false
    transform!(a, get_parameters!(a, args))
    if a.integral == :none
        return any(eachindex(a.x_val_scaled)) do i
            return a.eval_mask[i] && !covered(a, a.x_val_scaled[i], a.curve[i])
        end
    else
        # The data of each curve span an interval, so covering both ends of an
        # integration range covers all of it.
        return any(1:n_curves(a)) do c
            r = integration_range(a, c)
            return !isnothing(r) && !(covered(a, r[1], c) && covered(a, r[2], c))
        end
    end
end

"""
    extrapolation_free_window(a::Autoscaler, params)

At the full parameters `params`, the number of scored points that need extrapolation, and
the widest part of `eval_window` around its centre in which no point does (or `nothing`).
A point needs extrapolation when its scaled `x` lies outside the data of another curve.
"""
function extrapolation_free_window(a::Autoscaler, params)
    transform!(a, params)
    ok = [covered(a, a.x_val_scaled[i], a.curve[i]) for i in eachindex(a.x_val_scaled)]
    n_extrapolated = count(.!ok .& a.eval_mask)
    # Rows that share an unscaled `x` are kept or dropped together by a window.
    candidates = sort!(unique(raw_x(a)[a.eval_mask]))
    bad = Set(raw_x(a)[.!ok])
    centre = sum(a.eval_window) / 2
    k = argmin(abs.(candidates .- centre))
    if candidates[k] in bad
        return n_extrapolated, nothing
    end
    lo = hi = k
    while lo > 1 && candidates[lo - 1] ∉ bad
        lo -= 1
    end
    while hi < length(candidates) && candidates[hi + 1] ∉ bad
        hi += 1
    end
    return n_extrapolated, (candidates[lo], candidates[hi])
end

function warn_extrapolation(a::Autoscaler, args)
    extrapolates(a, args) || return nothing
    n, window = extrapolation_free_window(a, get_parameters(a, args))
    if isnothing(window)
        suggestion = "no part of `eval_window` around its centre is free of it"
    else
        suggestion = "`eval_window=$window` would need none"
    end
    if n > 0
        location = "at $n of $(count(a.eval_mask)) scored points"
    else
        location = "near the edges of `eval_window`"
    end
    @warn "The fitted collapse extrapolates curves beyond their data $location, so its " *
          "cost compares them with extrapolated master curves; $suggestion."
    return nothing
end
