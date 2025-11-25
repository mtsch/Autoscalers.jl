"""


# Usage

```jldoctest

```
"""
struct Autoscaler{P,S<:ScalingAnsatz{P}}
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

    # TODO: remove me?
    selected_x_val::Vector{Float64}
    selected_y_val::Vector{Float64}
    selected_y_err::Vector{Float64}

    hardcodes::Vector{Float64}
    last_params::Vector{Float64}

    classic_cost_function::Bool
    use_all_sizes::Bool

    window_width::Float64
end
function Autoscaler(ansatz::ScalingAnsatz=DEFAULT_ANSATZ; x, y, y_err, L, kwargs...)
    return Autoscaler(x, y, y_err, size)
end
function Autoscaler(
    table, ansatz::ScalingAnsatz=DEFAULT_ANSATZ;
    x=:x, y=:y, y_err=:y_err, L=:L,
    classic_cost_function=false,
    use_all_sizes=false,
    window_width=1,
    kwargs...
)

    rows = collect(Tables.rows(table))
    sort!(rows, by=r -> (r[L], r[x]))

    x_val_raw = Float64[]
    y_val_raw = Float64[]
    y_err_raw = Float64[]
    sizes = Int[]
    avail_sizes = Int[]
    size_map = Dict{Int,UnitRange{Int}}()

    for (k, _) in kwargs
        if k ∉ ansatz.param_names
            throw(ArgumentError("unrecognized keyword argument `$k`"))
        end
    end

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

    result = Autoscaler(
        ansatz,
        x_val_raw, y_val_raw, y_err_raw,
        copy(x_val_raw), copy(y_val_raw), copy(y_err_raw),
        sizes, avail_sizes, size_map,
        Float64[], Float64[], Float64[],
        fill(NaN, length(ansatz.param_names)), zeros(length(ansatz.param_names)),
        classic_cost_function, use_all_sizes, Float64(window_width),
    )
    return hardcode_params!(result; kwargs...)
end

function hardcode_params!(c::Autoscaler; kwargs...)
    param_names = c.ansatz.param_names
    hardcodes = c.hardcodes
    for (i, p) in enumerate(param_names)
        hardcodes[i] = get(kwargs, p, NaN)
    end
    return c
end

function Base.show(io::IO, c::Autoscaler)
    print(io, "AutoScaler(", c.ansatz, "...)")
end

function transform!(c::Autoscaler, params)
    for i in eachindex(c.x_val_scaled)
        L, x, y, y_err = c.sizes[i], c.x_val_raw[i], c.y_val_raw[i], c.y_err_raw[i]

        c.x_val_scaled[i] = c.ansatz.scale_x(x, L, params)

        y_measurement_scaled = c.ansatz.scale_y(y ± y_err, L, params)
        c.y_val_scaled[i] = y_measurement_scaled.val
        c.y_err_scaled[i] = y_measurement_scaled.err
    end
    return nothing
end

function scaled_data(c::Autoscaler, size)
    range = c.size_map[size]
    return (
        x_val = view(c.x_val_scaled, range),
        y_val = view(c.y_val_scaled, range),
        y_err = view(c.y_err_scaled, range),
    )
end

function window(a::Autoscaler)
    lo = -Inf
    hi = Inf
    for size in a.avail_sizes
        x_scaled = scaled_data(a, size)[1]
        lo = max(lo, first(x_scaled))
        hi = min(hi, last(x_scaled))
    end
    if lo > hi
        return lo, hi
    end
    if a.window_width ≠ 1
        window = hi - lo
        desired_window = window * a.window_width
        to_shrink = window - desired_window

        # shrink the larger part first
        diff = abs(hi) - abs(lo)
        if diff < 0
            lo += min(-diff, to_shrink)
            to_shrink -= min(-diff, to_shrink)
        else
            hi -= min(diff, to_shrink)
            to_shrink -= min(diff, to_shrink)
        end

        lo += to_shrink / 2
        hi -= to_shrink / 2
        @assert hi - lo ≈ desired_window
    end

    return lo, hi
end

"""
    get_parameters(c::Autoscaler{N}, args)

Get the parameters `x_crit`, `a`, `b` taking hardcoded values into account. `args` must be
of an appropriate length.
"""
function get_parameters(c::Autoscaler, args)
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

function (c::Autoscaler)(params::Vector)
    params = get_parameters(c, params)
    return cost_function(c, params)
end
function (c::Autoscaler)(args::Vararg{<:Real})
    params = get_parameters(c, args)
    return cost_function(c, params)
end

function to_table(c::Autoscaler, args::Vararg{<:Real})
    return to_table(c, args)
end
function to_table(c::Autoscaler, args::Union{Vector,Tuple})
    params = get_parameters(c, args)
    transform!(c, params)
    if c.classic_cost_function
        master_curve = [
            standard_master_curve(c, x, size) for (x, size) in zip(c.x_val_scaled, c.sizes)
        ]
    else
        master_curve = [
            new_master_curve(c, x, size) for (x, size) in zip(c.x_val_scaled, c.sizes)
        ]
    end
    return (;
        L=copy(c.sizes),
        x=copy(c.x_val_scaled),
        y=copy(c.y_val_scaled),
        y_err=copy(c.y_err_scaled),
        master_curve,
    )
end



function cost_function(c::Autoscaler, params)
    if c.classic_cost_function
        return standard_cost_function(c, params)
    else
        return new_cost_function(c, params)
        #return better_cost_function(c, params)
    end
end
