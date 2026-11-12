# Fixed-window cost function: a fixed set of points, chosen in raw `x`, is compared to the
# master curve of the other curves. Extrapolation beyond a curve's data is penalised
# smoothly, so the cost is continuous in the parameters and cannot be lowered by moving the
# window. Extrapolated curves carry the propagated, growing variance of the extrapolation,
# which on its own would reward extrapolating further; the penalty or the log-variance term
# of the likelihood (`log_likelihood=true`) counters that.

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

Fixed-window cost, split into the collapse (χ²-like) part, the extrapolation penalty and,
with `a.log_likelihood`, the log-variance term of the Gaussian likelihood. Each curve
contributes the mean over its scored points; the result is averaged over curves with at
least one scored point.
"""
function fixed_window_breakdown(a::Autoscaler, params)
    transform!(a, params)
    chi2_total = pen_total = logvar_total = 0.0
    n_scored = 0
    for (c, range) in enumerate(a.curve_ranges)
        chi2 = pen = logvar = 0.0
        n = 0
        for i in range
            a.eval_mask[i] || continue
            m = fixed_master_curve(a, a.x_val_scaled[i], c)
            σ2 = a.y_err_scaled[i]^2
            chi2 += (a.y_val_scaled[i] - m.Y)^2 / (σ2 + m.dY2)
            pen += m.penalty
            # log(σ² + σ_Y²) up to the constant log σ², which does not depend on the
            # parameters: zero for an exact master curve.
            logvar += log1p(m.dY2 / σ2)
            n += 1
        end
        n == 0 && continue
        chi2_total += chi2 / n
        pen_total += pen / n
        logvar_total += logvar / n
        n_scored += 1
    end
    logvar = a.log_likelihood ? logvar_total / n_scored : 0.0
    return (;
        cost=(chi2_total + pen_total) / n_scored + logvar,
        chi2=chi2_total / n_scored,
        penalty=pen_total / n_scored,
        log_variance=logvar,
    )
end

fixed_window_cost(a::Autoscaler, params) = fixed_window_breakdown(a, params).cost

"""
    cost_breakdown(a::Autoscaler, args...)

The fixed-window cost at the free parameters `args`, split into its parts: a `NamedTuple`
`(; cost, chi2, penalty, log_variance)` with `cost = chi2 + penalty + log_variance`.
Requires `eval_window`.

* `chi2` is the χ² per scored point, a measure of goodness of fit also when the cost is a
  likelihood.
* A non-negligible `penalty` means the collapse relies on extrapolating curves beyond their
  data; shrink `eval_window` until it vanishes.
* `log_variance` is the mean of ``\\log(1 + σ_Y^2/σ^2)`` over the scored points with
  `log_likelihood=true`, and zero otherwise.

# Examples

```jldoctest
julia> data = [(; L, x, y=L^-0.25 * tanh(L * (x - 0.3)), y_err=1e-3)
               for L in (8, 16, 32) for x in 0.2:0.005:0.4];

julia> s = Autoscaler(data; eval_window=(0.28, 0.32));

julia> map(c -> round(c; digits=3), cost_breakdown(s, 1.0, 0.25, 0.3))
(cost = 0.005, chi2 = 0.005, penalty = 0.0, log_variance = 0.0)
```
"""
function cost_breakdown(a::Autoscaler, args::Union{AbstractVector,Tuple})
    isnothing(a.eval_window) && throw(ArgumentError("`eval_window` is not set"))
    return fixed_window_breakdown(a, get_parameters!(a, args))
end
cost_breakdown(a::Autoscaler, args::Vararg{Real}) = cost_breakdown(a, args)

"""
    extrapolates(a::Autoscaler, args)

Whether the fixed-window cost at the free parameters `args` relies on extrapolating curves
beyond their data, whatever the penalty. Always `false` for the classic cost.
"""
function extrapolates(a::Autoscaler, args)
    isnothing(a.eval_window) && return false
    return first(extrapolation_free_window(a, get_parameters(a, args))) > 0
end

"""
    extrapolation_free_window(a::Autoscaler, params)

At the full parameters `params`, the number of scored points that need extrapolation, and
the widest part of `eval_window` around its centre in which no point does (or `nothing`).
A point needs extrapolation when its scaled `x` lies outside the data of another curve.
"""
function extrapolation_free_window(a::Autoscaler, params)
    transform!(a, params)
    covered = map(eachindex(a.x_val_scaled)) do i
        x = a.x_val_scaled[i]
        return all(1:n_curves(a)) do c
            xs = scaled_data(a, c).x_val
            return c == a.curve[i] || xs[1] ≤ x ≤ xs[end]
        end
    end
    n_extrapolated = count(.!covered .& a.eval_mask)
    # Rows that share an unscaled `x` are kept or dropped together by a window.
    candidates = sort!(unique(raw_x(a)[a.eval_mask]))
    bad = Set(raw_x(a)[.!covered])
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
    @warn "The fitted collapse extrapolates curves beyond their data at $n of " *
          "$(count(a.eval_mask)) scored points; $suggestion."
    return nothing
end
