# Classic cost function of Houdayer & Hartmann, PRB 70, 014418 (2004): every point is
# compared to a weighted linear fit through the neighbouring points of all other sizes.

function select_subset(c::Autoscaler, selected_x, selected_size)
    empty!(c.selected_x_val)
    empty!(c.selected_y_val)
    empty!(c.selected_y_err)
    for size in c.avail_sizes
        size == selected_size && continue
        x, y, y_err = scaled_data(c, size)
        for i in eachindex(x)
            isnan(x[i]) && continue
            if x[i] > selected_x
                if i > 1 && !isnan(x[i-1])
                    append!(c.selected_x_val, (x[i-1], x[i]))
                    append!(c.selected_y_val, (y[i-1], y[i]))
                    append!(c.selected_y_err, (y_err[i-1], y_err[i]))
                end
                break
            end
        end
    end
    return c.selected_x_val, c.selected_y_val, c.selected_y_err
end

"""
    standard_master_curve(c::Autoscaler, selected_x, selected_size)

Weighted linear fit through the neighbours of `selected_x` in all other sizes, evaluated at
`selected_x`. Returns `(Y, dY2)` or `missing` if no other size brackets `selected_x`.
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
