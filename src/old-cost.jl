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

function standard_master_curve(c::Autoscaler, selected_x, selected_size)
    x, y, y_err = select_subset(c, selected_x, selected_size)
    if isempty(x)
        return missing
    else
        y_err .= 1 ./ y_err .^ 2
        weights = y_err
        indices = eachindex(weights)

        K = sum(weights[i] for i in indices)
        Kx = sum(weights[i] * x[i] for i in indices)
        Ky = sum(weights[i] * y[i] for i in indices)
        Kxx = sum(weights[i] * x[i]^2 for i in indices)
        Kxy = sum(weights[i] * x[i] * y[i] for i in indices)
        Δ = K * Kxx - Kx^2

        Y = (Kxx * Ky - Kx * Kxy) / Δ + selected_x * (K * Kxy - Kx * Ky) / Δ
        dY2 = (Kxx - 2selected_x * Kx + selected_x^2 * K) / Δ

        return Y, dY2
    end
end

function standard_cost_function(c::Autoscaler, params)
    transform!(c, params)
    s = 0.0
    n = 0
    for i in eachindex(c.x_val_raw)
        master = standard_master_curve(c, c.x_val_scaled[i], c.sizes[i])
        if !ismissing(master)
            Y, dY2 = master
            s += (c.y_val_raw[i] - Y)^2 / (c.y_err_raw[i]^2 + dY2)
            n += 1
        end
    end
    return s / n
end
