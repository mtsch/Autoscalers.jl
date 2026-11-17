# Experimental windowed integral cost. For each curve, the squared deviation from the master
# curve of the other curves is integrated over the part of `eval_window` (in unscaled x)
# that the curve's own data cover, and normalised by the length of that part.

"""
    interpolate_variance(xs, ys, dys, x, exact)

Like `extrapolate`, but between knots the variance is either interpolated linearly in σ²
(`exact=false`) or is the variance of the linear interpolant of independent points,
``(1-w)^2 σ_1^2 + w^2 σ_2^2`` (`exact=true`). Outside the data it is `extrapolate`.
"""
function interpolate_variance(xs, ys, dys, x, exact)
    if !exact || x < xs[1] || x > xs[end]
        return extrapolate(xs, ys, dys, x)
    end
    k = searchsortedlast(xs, x)
    k == length(xs) && return ys[k], dys[k]^2, 0.0
    w = (x - xs[k]) / (xs[k + 1] - xs[k])
    y = (1 - w) * ys[k] + w * ys[k + 1]
    return y, (1 - w)^2 * dys[k]^2 + w^2 * dys[k + 1]^2, 0.0
end

# Integrand at scaled `x` for curve `c`: (χ² density, penalty density).
function integrand(a::Autoscaler, c, x, exact)
    num = den = pen = 0.0
    for c′ in 1:n_curves(a)
        c′ == c && continue
        xs, ys, dys = scaled_data(a, c′)
        y, dy2, d = interpolate_variance(xs, ys, dys, x, exact)
        num += y / dy2
        den += 1 / dy2
        pen += a.penalty * d^2
    end
    xs, ys, dys = scaled_data(a, c)
    y, dy2, _ = interpolate_variance(xs, ys, dys, x, exact)
    return (y - num / den)^2 / (dy2 + 1 / den), pen
end

# Scaled x at unscaled `t` on curve `c`, interpolated between the curve's own points (exact
# when the scaling map is affine in t, as for L^a (t - t_c)).
function scaled_at(a::Autoscaler, c, t)
    range = a.curve_ranges[c]
    ts = view(raw_x(a), range)
    xs = view(a.x_val_scaled, range)
    k = clamp(searchsortedlast(ts, t), 1, length(ts) - 1)
    return lerp((ts[k], xs[k]), (ts[k + 1], xs[k + 1]), t)
end

function integral_window_breakdown(a::Autoscaler, params)
    transform!(a, params)
    exact = a.integral == :exact
    lo, hi = a.eval_window
    chi2_total = pen_total = 0.0
    n_scored = 0
    knots = Float64[]
    for (c, range) in enumerate(a.curve_ranges)
        ts = view(raw_x(a), range)
        t1, t2 = max(lo, ts[1]), min(hi, ts[end])
        t2 > t1 || continue
        x1, x2 = scaled_at(a, c, t1), scaled_at(a, c, t2)
        empty!(knots)
        push!(knots, x1, x2)
        for x in a.x_val_scaled
            x1 < x < x2 && push!(knots, x)
        end
        sort!(knots)
        if a.integral in (:gk, :gk_exact)
            # Between knots every interpolant is linear, so the integrand is smooth there
            # and each segment is integrated separately; the kinks sit at the segment ends.
            # With the linear variance the integrand is nearly polynomial and a low order
            # suffices; the exact variance puts poles near the segment, where a higher order
            # is cheaper.
            var_exact = a.integral == :gk_exact
            order = var_exact ? 7 : 2
            gk(f) = sum(quadgk(f, knots[k - 1], knots[k]; order, rtol=1e-6)[1]
                        for k in 2:length(knots))
            chi2_total += gk(x -> integrand(a, c, x, var_exact)[1]) / (x2 - x1)
            if a.penalty > 0
                pen_total += gk(x -> integrand(a, c, x, var_exact)[2]) / (x2 - x1)
            end
            n_scored += 1
            continue
        end
        chi2 = pen = 0.0
        prev = integrand(a, c, knots[1], exact)
        for k in 2:length(knots)
            curr = integrand(a, c, knots[k], exact)
            h = (knots[k] - knots[k - 1]) / 2
            chi2 += h * (prev[1] + curr[1])
            pen += h * (prev[2] + curr[2])
            prev = curr
        end
        chi2_total += chi2 / (x2 - x1)
        pen_total += pen / (x2 - x1)
        n_scored += 1
    end
    return (;
        cost=(chi2_total + pen_total) / n_scored,
        chi2=chi2_total / n_scored,
        penalty=pen_total / n_scored,
    )
end

integral_window_cost(a::Autoscaler, params) = integral_window_breakdown(a, params).cost
