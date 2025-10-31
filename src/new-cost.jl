function deviation_at(a::Autoscaler, x, selected_size, selected_curve)
    m = better_master_curve(a, x, selected_size)
    ismissing(m) && return missing

    s = interpolate(selected_curve..., x; sq_errs=true)
    ismissing(s) && return missing

    y, dy2 = s
    Y, dY2 = m

    δ = (y - Y)^2 / (dy2 + dY2)
    return δ
end

function partial_cost_function(a::Autoscaler, selected_size, (lo, hi)=window(a))
    selected_curve = scaled_data(a, selected_size)

    res = 0.0
    xs = range(lo, hi; length=10_000)
    x_prev = first(xs)
    y_prev, dy2_prev = interpolate(selected_curve..., x_prev; sq_errs=true)
    Y_prev, dY2_prev = better_master_curve(a, x_prev, selected_size)
    δ2_prev = (y_prev - Y_prev)^2 / (dy2_prev + dY2_prev)
    for i in 2:length(xs)
        x_curr = xs[i]
        y_curr, dy2_curr = interpolate(selected_curve..., x_curr; sq_errs=true)
        Y_curr, dY2_curr = better_master_curve(a, x_curr, selected_size)

        δ2_curr = (y_curr - Y_curr)^2 / (dy2_curr + dY2_curr)

        res += (x_curr - x_prev) * (δ2_curr + δ2_prev) / 2

        δ2_prev = δ2_curr
        x_prev = x_curr
    end

    return res / (hi - lo)
end

function new_cost_function(a::Autoscaler, params)
    transform!(a, params)
    lo, hi = window(a)

    if lo > hi
        return 1e12 * (lo - hi)
    end
    if hi - lo == 0.0
        return Inf
    end

    res = 0.0
    for size in a.avail_sizes
        res += partial_cost_function(a, size, (lo, hi))
    end
    return √(res / length(a.avail_sizes))
end
