# Fixed-window cost function: a fixed set of points, chosen in raw `x`, is compared to the
# master curve of the other curves. Extrapolation beyond a curve's data is penalised
# smoothly, so the cost is continuous in the parameters and cannot be lowered by moving the
# window.

"""
    fixed_master_curve(a::Autoscaler, x, own_curve)

Error-weighted master curve at scaled `x`, built from all curves except `own_curve`
(unless `a.use_all_sizes`), extrapolating where needed. Returns `(; Y, dY2, penalty)`.
"""
function fixed_master_curve(a::Autoscaler, x, own_curve)
    num = den = pen = 0.0
    for c in 1:n_curves(a)
        (!a.use_all_sizes && c == own_curve) && continue
        xs, ys, dys = scaled_data(a, c)
        y, dy2, d = extrapolate(xs, ys, dys, x)
        w = 1 / dy2
        num += w * y
        den += w
        pen += a.penalty * d^2
    end
    return (; Y=num / den, dY2=1 / den, penalty=pen)
end

"""
    fixed_window_breakdown(a::Autoscaler, params)

Fixed-window cost, split into the collapse (χ²-like) part and the extrapolation penalty.
Each curve contributes the mean over its scored points; the result is averaged over curves
with at least one scored point.
"""
function fixed_window_breakdown(a::Autoscaler, params)
    transform!(a, params)
    chi2_total = pen_total = 0.0
    n_scored = 0
    for (c, range) in enumerate(a.curve_ranges)
        chi2 = pen = 0.0
        n = 0
        for i in range
            a.eval_mask[i] || continue
            m = fixed_master_curve(a, a.x_val_scaled[i], c)
            chi2 += (a.y_val_scaled[i] - m.Y)^2 / (a.y_err_scaled[i]^2 + m.dY2)
            pen += m.penalty
            n += 1
        end
        n == 0 && continue
        chi2_total += chi2 / n
        pen_total += pen / n
        n_scored += 1
    end
    return (;
        cost=(chi2_total + pen_total) / n_scored,
        chi2=chi2_total / n_scored,
        penalty=pen_total / n_scored,
    )
end

fixed_window_cost(a::Autoscaler, params) = fixed_window_breakdown(a, params).cost

"""
    cost_breakdown(a::Autoscaler, args...)

Return `(; cost, chi2, penalty)` at the free parameters `args` (same convention as calling
the `Autoscaler`). Requires `eval_window`. A non-negligible `penalty` means the fit relies on
extrapolated curves; shrink `eval_window`.
"""
function cost_breakdown(a::Autoscaler, args::Union{AbstractVector,Tuple})
    isnothing(a.eval_window) && throw(ArgumentError("`eval_window` is not set"))
    return fixed_window_breakdown(a, get_parameters!(a, args))
end
cost_breakdown(a::Autoscaler, args::Vararg{Real}) = cost_breakdown(a, args)
