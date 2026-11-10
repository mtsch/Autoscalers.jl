# Classic cost function of Houdayer & Hartmann, PRB 70, 014418 (2004), Appendix A: every
# point is compared to a weighted linear fit through the neighbouring points of all other
# sizes.

"""
    select_subset(c::Autoscaler, selected_x, selected_size)

From every size other than `selected_size`, select the two neighbouring points `j, j + 1`
with `x[j] ≤ selected_x ≤ x[j + 1]`, or nothing if that size does not bracket `selected_x`
(Houdayer & Hartmann, Appendix A).
"""
function select_subset(c::Autoscaler, selected_x, selected_size)
    empty!(c.selected_x_val)
    empty!(c.selected_y_val)
    empty!(c.selected_y_err)
    for size in c.avail_sizes
        size == selected_size && continue
        x, y, y_err = scaled_data(c, size)
        n = length(x)
        j = searchsortedlast(x, selected_x)
        # On an interior knot both adjacent pairs bracket `selected_x`; the pair starting at
        # the knot matches the reference implementation (autoScale). On the last knot only
        # the pair ending there does.
        if j == n && x[n] == selected_x
            j = n - 1
        elseif j == 0 || j == n
            continue
        end
        append!(c.selected_x_val, (x[j], x[j + 1]))
        append!(c.selected_y_val, (y[j], y[j + 1]))
        append!(c.selected_y_err, (y_err[j], y_err[j + 1]))
    end
    return c.selected_x_val, c.selected_y_val, c.selected_y_err
end

"""
    standard_master_curve(c::Autoscaler, selected_x, selected_size)

Weighted linear fit through the neighbours of `selected_x` in all other sizes, evaluated at
`selected_x` (Eqs. A2 and A3). Returns `(Y, dY2)` or `missing` if no other size brackets
`selected_x`.
"""
function standard_master_curve(c::Autoscaler, selected_x, selected_size)
    x, y, y_err = select_subset(c, selected_x, selected_size)
    isempty(x) && return missing

    weights = y_err
    weights .= 1 ./ y_err .^ 2

    K = sum(weights)
    Kx = sum(weights[i] * x[i] for i in eachindex(weights))
    Ky = sum(weights[i] * y[i] for i in eachindex(weights))
    Kxx = sum(weights[i] * x[i]^2 for i in eachindex(weights))
    Kxy = sum(weights[i] * x[i] * y[i] for i in eachindex(weights))
    Δ = K * Kxx - Kx^2

    Y = (Kxx * Ky - Kx * Kxy) / Δ + selected_x * (K * Kxy - Kx * Ky) / Δ
    dY2 = (Kxx - 2selected_x * Kx + selected_x^2 * K) / Δ

    return Y, dY2
end

"""
    standard_cost_function(c::Autoscaler, params)

Quality `S` of Eq. A1: the mean of `(y - Y)^2 / (dy^2 + dY^2)` over all points for which the
master curve is defined. `Inf` if it is defined for none.
"""
function standard_cost_function(c::Autoscaler, params)
    transform!(c, params)
    s = 0.0
    n = 0
    for i in eachindex(c.x_val_scaled)
        master = standard_master_curve(c, c.x_val_scaled[i], c.sizes[i])
        if !ismissing(master)
            Y, dY2 = master
            s += (c.y_val_scaled[i] - Y)^2 / (c.y_err_scaled[i]^2 + dY2)
            n += 1
        end
    end
    return n == 0 ? Inf : s / n
end
