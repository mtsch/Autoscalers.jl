module Autoscaler

using Tables
export AutoscaleCostFunction, to_table

# TODO: only above/ only-below
struct AutoscaleCostFunction
    x_values::Vector{Float64}
    x_scaled::Vector{Float64}
    y_values::Vector{Float64}
    y_errors::Vector{Float64}
    sizes::Vector{Int}

    avail_sizes::Vector{Int}
    size_map::Dict{Int,UnitRange{Int}}

    selected_x_values::Vector{Float64}
    selected_y_values::Vector{Float64}
    selected_y_errors::Vector{Float64}

    side::Symbol
end
function AutoscaleCostFunction(side=:both; x, y, dy, l)
    return AutoscaleCostFunction(x, y, dy, l, side)
end
function AutoscaleCostFunction(x, y, dy, l, side=:both)
    return AutoscaleCostFunction((; x, y, dy, l), side)
end
function AutoscaleCostFunction(table, side=:both; x=:x, y=:y, dy=:dy, size=:l)
    if side ∉ (:both, :above, :below)
        throw(ArgumentError("`side` can only be `:both`, `:above`, or `:below`"))
    end

    rows = collect(Tables.rows(table))
    sort!(rows, by=r -> (r[size], r[x]))

    len = length(rows)
    x_values = Float64[]
    x_scaled = Float64[]
    y_values = Float64[]
    y_errors = Float64[]
    sizes = Int[]
    avail_sizes = Int[]
    size_map = Dict{Int,UnitRange{Int}}()

    prev_size = -1
    start_index = 0
    i = 0
    for row in rows
        if iszero(row[dy])
            @warn "Some data points have zero errors. Skipping." maxlog=1
            continue
        end
        i += 1

        curr_size = row[size]
        if curr_size ≠ prev_size
            if prev_size > 0
                size_map[prev_size] = start_index:(i - 1)
            end
            push!(avail_sizes, curr_size)
            prev_size = curr_size
            start_index = i
        end
        push!(x_values, row[x])
        push!(x_scaled, row[x])
        push!(y_values, row[y])
        push!(y_errors, row[dy])
        push!(sizes, row[size])
    end
    size_map[prev_size] = start_index:len

    return AutoscaleCostFunction(
        x_values, x_scaled, y_values, y_errors, sizes, avail_sizes, size_map,
        Float64[], Float64[], Float64[], side,
    )
end

function to_table(c::AutoscaleCostFunction, x_crit, ν)
    transform_x!(c, x_crit, ν)
    return (; size=c.sizes, x=c.x_scaled, y=c.y_values, dy=c.y_errors)
end

function transform_x!(c::AutoscaleCostFunction, x_crit, ν)
    c.x_scaled .= c.x_values .- x_crit
    if c.side == :below
        c.x_scaled[c.x_scaled ≥ 0.0] .= NaN
    elseif c.side == :above
        c.x_scaled[c.x_scaled ≤ 0.0] .= NaN
    end
    c.x_scaled .= c.sizes .* sign.(c.x_scaled) .* abs.(c.x_scaled) .^ ν
end

function scaled_data(c::AutoscaleCostFunction, size)
    range = c.size_map[size]
    return view(c.x_scaled, range), view(c.y_values, range), view(c.y_errors, range)
end

function select_subset(c::AutoscaleCostFunction, selected_x, selected_size)
    empty!(c.selected_x_values)
    empty!(c.selected_y_values)
    empty!(c.selected_y_errors)
    for size in c.avail_sizes
        size == selected_size && continue
        x, y, dy = scaled_data(c, size)
        for i in eachindex(x)
            if x[i] > selected_x
                if i > 1
                    append!(c.selected_x_values, (x[i-1], x[i]))
                    append!(c.selected_y_values, (y[i-1], y[i]))
                    append!(c.selected_y_errors, (dy[i-1], dy[i]))
                end
                break
            end
        end
    end
    return c.selected_x_values, c.selected_y_values, c.selected_y_errors
end

function master_curve_at(c::AutoscaleCostFunction, selected_x, selected_size)
    x, y, dy = select_subset(c, selected_x, selected_size)
    if isempty(x)
        return missing
    else
        dy .= 1 ./ dy .^ 2
        weights = dy
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


function (c::AutoscaleCostFunction)(params::Vector)
    return c(params...)
end
function (c::AutoscaleCostFunction)(x_crit, ν)
    transform_x!(c, x_crit, ν)

    s = 0.0
    n = 0
    for i in eachindex(c.x_values)
        master = master_curve_at(c, c.x_scaled[i], c.sizes[i])
        if !ismissing(master)
            Y, dY2 = master
            s += (c.y_values[i] - Y)^2 / (c.y_errors[i]^2 + dY2)
            n += 1
        end
    end
    return s / n
end

struct PartiallyAppliedAutoscaleCostFunction
    cost_fun::AutoscaleCostFunction
    x_crit::Float64
end
function (c::AutoscaleCostFunction)(x_crit)
    return PartiallyAppliedAutoscaleCostFunction(c, x_crit)
end
function (c::PartiallyAppliedAutoscaleCostFunction)(param::Vector)
    return c.cost_fun(c.x_crit, param[1])
end
function (c::PartiallyAppliedAutoscaleCostFunction)(ν)
    return c.cost_fun(c.x_crit, ν)
end


end
