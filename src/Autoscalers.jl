module Autoscalers

using Tables
using Measurements
export Autoscaler, to_table, @ansatz

include("ansatz.jl")



"""


# Usage

```jldoctest

```
"""
struct Autoscaler4{N,S<:ScalingAnsatz{N}}
    ansatz::S

    x_val_raw::Vector{Float64}
    y_val_raw::Vector{Float64}
    y_err_raw::Vector{Float64}

    x_val_scaled::Vector{Float64}
    y_val_scaled::Vector{Float64}
    y_err_scaled::Vector{Float64}

    sizes::Vector{Int}

    avail_sizes::Vector{Int}
    size_map::Dict{Int,UnitRange{Int}}

    selected_x_val::Vector{Float64}
    selected_y_val::Vector{Float64}
    selected_y_err::Vector{Float64}

    hardcodes::Vector{Float64}
    last_params::Vector{Float64}
end
function Autoscaler4(ansatz::ScalingAnsatz=DEFAULT_ANSATZ; x, y, y_err, L, kwargs...)
    return Autoscaler4(x, y, y_err, size)
end
function Autoscaler4(
    table, ansatz::ScalingAnsatz=DEFAULT_ANSATZ;
    x=:x, y=:y, y_err=:y_err, L=:L, kwargs...
)

    rows = collect(Tables.rows(table))
    sort!(rows, by=r -> (r[L], r[x]))

    x_val_raw = Float64[]
    y_val_raw = Float64[]
    y_err_raw = Float64[]
    sizes = Int[]
    avail_sizes = Int[]
    size_map = Dict{Int,UnitRange{Int}}()

    prev_size = -1
    start_index = 0
    i = 0
    for row in rows
        if iszero(row[y_err])
            @warn "Some data points have zero errors. Skipping." maxlog=1
            continue
        end
        i += 1

        curr_size = row[L]
        if curr_size ≠ prev_size
            if prev_size > 0
                size_map[prev_size] = start_index:(i - 1)
            end
            push!(avail_sizes, curr_size)
            prev_size = curr_size
            start_index = i
        end
        push!(x_val_raw, row[x])
        push!(y_val_raw, row[y])
        push!(y_err_raw, row[y_err])
        push!(sizes, row[L])
    end
    size_map[prev_size] = start_index:length(x_val_raw)

    result = Autoscaler4(
        ansatz,
        x_val_raw, y_val_raw, y_err_raw,
        copy(x_val_raw), copy(y_val_raw), copy(y_err_raw),
        sizes, avail_sizes, size_map,
        Float64[], Float64[], Float64[],
        fill(NaN, length(ansatz.param_names)), zeros(length(ansatz.param_names)),
    )
    return hardcode_params!(result; kwargs...)
end

function hardcode_params!(c::Autoscaler4; kwargs...)
    param_names = c.ansatz.param_names
    hardcodes = c.hardcodes
    for (i, p) in enumerate(param_names)
        hardcodes[i] = get(kwargs, p, NaN)
    end
    return c
end

function Base.show(io::IO, c::Autoscaler4)
    print(io, "AutoScaler(", c.ansatz, "...)")
end

function transform!(c::Autoscaler4, params)
    for i in eachindex(c.x_val_scaled)
        L, x, y, y_err = c.sizes[i], c.x_val_raw[i], c.y_val_raw[i], c.y_err_raw[i]

        c.x_val_scaled[i] = c.ansatz.scale_x(x, L, params)

        y_measurement_scaled = c.ansatz.scale_y(y ± y_err, L, params)
        c.y_val_scaled[i] = y_measurement_scaled.val
        c.y_err_scaled[i] = y_measurement_scaled.err
    end
    return nothing
end

function scaled_data(c::Autoscaler4, size)
    range = c.size_map[size]
    return (
        view(c.x_val_scaled, range),
        view(c.y_val_scaled, range),
        view(c.y_err_scaled, range),
    )
end

function select_subset(c::Autoscaler4, selected_x, selected_size)
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

function master_curve_at(c::Autoscaler4, selected_x, selected_size)
    return improved_master_curve_at(c, selected_x, selected_size)
end
function classic_master_curve_at(c::Autoscaler4, selected_x, selected_size)
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
function lerp(left, right, selected_x)
    x1, y1 = left
    x2, y2 = right
    return (y1 * (x2 - selected_x) + y2 * (selected_x - x1)) / (x2 - x1)
end
function improved_master_curve_at(c::Autoscaler4, selected_x, selected_size)
    x, y, y_err = select_subset(c, selected_x, selected_size)
    if isempty(x)
        return missing
    else
        mean_top = 0.0
        mean_bot = 0.0
        for i in 1:2:length(x)
            y_val = lerp((x[i], y[i]), (x[i+1], y[i+1]), selected_x)
            y_err2 = lerp((x[i], y_err[i]^2), (x[i+1], y_err[i+1]^2), selected_x)

            weight = 1/y_err2
            mean_top += y_val * weight
            mean_bot += weight
        end

        Y = mean_top / mean_bot
        dY2 = 1 / mean_bot

        return Y, dY2
    end
end

function cost_function(c::Autoscaler4, params)
    return standard_cost_function(c, params)
end
function standard_cost_function(c::Autoscaler4, params)
    transform!(c, params)
    s = 0.0
    n = 0
    for i in eachindex(c.x_val_raw)
        master = master_curve_at(c, c.x_val_scaled[i], c.sizes[i])
        if !ismissing(master)
            Y, dY2 = master
            s += (c.y_val_raw[i] - Y)^2 / (c.y_err_raw[i]^2 + dY2)
            n += 1
        end
    end
    return s / n
end
function gaussian_cost_function(c::Autoscaler4, params, σ=0.01)
    transform!(c, params)
    s = 0.0
    n = 0.0
    for i in eachindex(c.x_val_raw)
        w = exp(-(c.x_val_raw[i] - x_crit)^2 / 2σ^2)
        master = master_curve_at(c, c.x_val_scaled[i], c.sizes[i])
        if !ismissing(master)
            Y, dY2 = master
            s += (c.y_val_raw[i] - Y)^2 / (c.y_err_raw[i]^2 + dY2) * w
            n += w
        end
    end
    return s / n
end


"""
    get_parameters(c::Autoscaler4{N}, args)

Get the parameters `x_crit`, `a`, `b` taking hardcoded values into account. `args` must be
of an appropriate length.
"""
function get_parameters(c::Autoscaler4, args)
    expected = count(isnan, c.hardcodes)
    if length(args) ≠ expected
        throw(ArgumentError("expected $(expected) params, got $(length(args))"))
    end

    params = c.last_params
    param_names = c.ansatz.param_names

    arg_idx = 1
    for i in eachindex(params)
        name = param_names[i]
        if isnan(c.hardcodes[i])
            params[i] = args[arg_idx]
            arg_idx += 1
        else
            params[i] = c.hardcodes[i]
        end
    end

    return params
end

function (c::Autoscaler4)(params::Vector)
    params = get_parameters(c, params)
    return cost_function(c, params)
end
function (c::Autoscaler4)(args::Vararg{<:Real})
    params = get_parameters(c, args)
    return cost_function(c, params)
end

function to_table(c::Autoscaler4, args::Vararg{<:Real})
    return to_table(c, args)
end
function to_table(c::Autoscaler4, args::Union{Vector,Tuple})
    params = get_parameters(c, args)
    transform!(c, params)
    master_curve = [
        master_curve_at(c, x, size) for (x, size) in zip(c.x_val_scaled, c.sizes)
    ]
    return (;
        L=copy(c.sizes),
        x=copy(c.x_val_scaled),
        y=copy(c.y_val_scaled),
        y_err=copy(c.y_err_scaled),
        master_curve,
    )
end

# TODO: custom scaling ansatz
Autoscaler = Autoscaler4

end
