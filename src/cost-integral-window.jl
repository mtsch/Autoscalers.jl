# Experimental windowed integral cost. For each curve, the χ² density of its linear
# interpolant against the master curve of the other curves is integrated over the part of
# `eval_window` (in unscaled x) that the curve's own data cover, and normalised by the
# length of that part in scaled x.

"""
    integration_range(a::Autoscaler, c)

The scaled `x` at the ends of the part of `eval_window` that the data of curve `c` cover,
or `nothing` if they cover none of it. Uses the current scaling of `a`.
"""
function integration_range(a::Autoscaler, c)
    lo, hi = a.eval_window
    range = a.curve_ranges[c]
    ts = view(raw_x(a), range)
    t1, t2 = max(lo, ts[1]), min(hi, ts[end])
    t2 > t1 || return nothing
    # The ends lie within the curve's data, so this never extrapolates. It is exact when
    # the scaling is affine in t, as for L^a (t - t_c).
    xs = view(a.x_val_scaled, range)
    return interpolate(ts, xs, t1)::Float64, interpolate(ts, xs, t2)::Float64
end

# χ² density of curve `c` at scaled `x`.
function integrand(a::Autoscaler, c, x, exact)
    m = fixed_master_curve(a, x, c; exact)
    y, dy2 = extrapolate(scaled_data(a, c)..., x; exact)
    return (y - m.Y)^2 / (dy2 + m.dY2)
end

"""
    integral_window_cost(a::Autoscaler, params)

Windowed integral cost: for each curve, the mean of its ``χ^2`` density over its
[`integration_range`](@ref), averaged over the curves that have one.
"""
function integral_window_cost(a::Autoscaler, params)
    transform!(a, params)
    exact = a.integral == :gk_exact
    # With the linear variance the integrand is nearly polynomial and a low order suffices;
    # the exact variance puts poles near each segment, where a higher order is cheaper.
    order = exact ? 7 : 2
    total = 0.0
    n_scored = 0
    knots = Float64[]
    for c in 1:n_curves(a)
        r = integration_range(a, c)
        isnothing(r) && continue
        x1, x2 = r
        empty!(knots)
        push!(knots, x1, x2)
        for x in a.x_val_scaled
            x1 < x < x2 && push!(knots, x)
        end
        sort!(knots)
        # Between knots every interpolant is linear, so the integrand is smooth there;
        # integrating each segment separately keeps its kinks at the segment ends.
        f(x) = integrand(a, c, x, exact)
        chi2 = sum(quadgk(f, knots[k - 1], knots[k]; order, rtol=1e-6)[1]
                   for k in 2:length(knots))
        total += chi2 / (x2 - x1)
        n_scored += 1
    end
    return total / n_scored
end
