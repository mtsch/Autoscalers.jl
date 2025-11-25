module Autoscalers

using Tables
using Measurements
export Autoscaler, to_table, @ansatz

include("ansatz.jl")
include("autoscaler.jl")
include("interpolated-curve.jl")

include("new-cost.jl")
include("old-cost.jl")

###
###
###
function better_master_curve(a::Autoscaler, selected_x, selected_size)
    numerator = 0.0
    denominator = 0.0

    for size in a.avail_sizes
        size == selected_size && continue
        interp = interpolate(scaled_data(a, size)..., selected_x; sq_errs=true)
        if ismissing(interp)
            return missing
        else
            y_val, y_err2 = interp
            weight = 1 / y_err2
            numerator += y_val * weight
            denominator += weight
        end
    end
    return numerator / denominator, 1 / denominator
end

function better_cost_function(a::Autoscaler, params)
    transform!(a, params)

    numerator = 0.0
    denominator = 0.0

   lo, hi = window(a)
    if lo ≥ hi
        return 1e9 + (lo - hi)
    elseif lo > 0
        return 1e9 + lo
    elseif hi < 0
        return 1e9 - hi
    end
    xs = range(lo, hi; length=length(a.x_val_raw))
    for size in a.avail_sizes
        curr_x, curr_y, curr_y_err = scaled_data(a, size)
        for x in curr_x
            lo ≤ x ≤ hi || continue
            y, y_err = interpolate(curr_x, curr_y, curr_y_err, x)
            Y, dY2 = better_master_curve(a, x, size)

            numerator += (y - Y)^2 / (y_err^2 + dY2)
            denominator += 1
        end
    end
    return numerator / denominator
end


end
