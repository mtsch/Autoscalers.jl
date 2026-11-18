# Experimental windowed integral cost. For each curve, the χ² density of its linear
# interpolant against the master curve of the other curves is integrated over the part of
# `eval_window` (in unscaled x) that the curve's own data cover, and normalised by the
# length of that part in scaled x.

"""
    raw_overlap(ts, window)

The part `(t1, t2)` of `window` covered by the sorted unscaled `x` values `ts` of a curve,
or `nothing` if it has no positive length.
"""
function raw_overlap(ts, (lo, hi))
    t1, t2 = max(lo, first(ts)), min(hi, last(ts))
    return t2 > t1 ? (t1, t2) : nothing
end

"""
    integration_range(a::Autoscaler, c)

The scaled `x` at the ends of the part of `eval_window` that the data of curve `c` cover
(see [`raw_overlap`](@ref)), or `nothing` if they cover none of it. Uses the current
scaling of `a`.
"""
function integration_range(a::Autoscaler, c)
    range = a.curve_ranges[c]
    ts = view(raw_x(a), range)
    overlap = raw_overlap(ts, a.eval_window)
    isnothing(overlap) && return nothing
    # The ends lie within the curve's data, so this never extrapolates. It is exact when
    # the scaling is affine in t, as for L^a (t - t_c).
    xs = view(a.x_val_scaled, range)
    t1, t2 = overlap
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
[`integration_range`](@ref), averaged over the curves that have one. The integral is
evaluated with adaptive Gauss–Kronrod quadrature from QuadGK.jl.
"""
function integral_window_cost(a::Autoscaler, params)
    transform!(a, params)
    exact = a.cost == :gk_exact
    # With the linear variance the integrand is nearly polynomial and a low order suffices;
    # the exact variance puts poles near each segment, where a higher order is cheaper.
    order = exact ? 7 : 2
    total = 0.0
    n_scored = 0
    knots = a.knots
    for c in 1:n_curves(a)
        r = integration_range(a, c)
        isnothing(r) && continue
        x1, x2 = r
        empty!(knots)
        push!(knots, x1, x2)
        for x in a.x_val_scaled
            x1 < x < x2 && push!(knots, x)
        end
        # Curves on a common grid share knots; a repeated knot would be an empty segment
        # that still costs QuadGK a full rule evaluation.
        unique!(sort!(knots))
        # Between knots every interpolant is linear, so the integrand is smooth there;
        # breaking the interval at the knots keeps its kinks at segment ends.
        f(x) = integrand(a, c, x, exact)
        chi2 = quadgk(f, knots; order, rtol=1e-6, segbuf=a.segbuf)[1]
        total += chi2 / (x2 - x1)
        n_scored += 1
    end
    return total / n_scored
end
