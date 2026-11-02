"""
    Autoscaler(
        table, ansatz::ScalingAnsatz=DEFAULT_ANSATZ;
        x=:x, y=:y, y_err=:y_err, L=:L,
        classic_cost_function=false,
        use_all_sizes=false,
        window_width=1,
        kwargs...
    )
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

function hardcode_params!(a::Autoscaler; kwargs...)
    param_names = a.ansatz.param_names
    hardcodes = a.hardcodes
    for (i, p) in enumerate(param_names)
        hardcodes[i] = get(kwargs, p, NaN)
    end
    return a
end

function Base.show(io::IO, a::Autoscaler)
    print(io, "AutoScaler(", a.ansatz, "...)")
end

function transform!(a::Autoscaler, params)
    for i in eachindex(a.x_val_scaled)
        L, x, y, y_err = a.sizes[i], a.x_val_raw[i], a.y_val_raw[i], a.y_err_raw[i]

        a.x_val_scaled[i] = a.ansatz.scale_x(x, L, params)

        y_measurement_scaled = a.ansatz.scale_y(x, y ± y_err, L, params)
        a.y_val_scaled[i] = y_measurement_scaled.val
        a.y_err_scaled[i] = y_measurement_scaled.err
    end
    return nothing
end

function scaled_data(a::Autoscaler, size)
    range = a.size_map[size]
    return (
        x_val = view(a.x_val_scaled, range),
        y_val = view(a.y_val_scaled, range),
        y_err = view(a.y_err_scaled, range),
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
    get_parameters(a::Autoscaler{N}, args)

Get the parameters `x_crit`, `a`, `b` taking hardcoded values into account. `args` must be
of an appropriate length.
"""
function get_parameters(a::Autoscaler, args)
    expected = count(isnan, a.hardcodes)
    if length(args) ≠ expected
        throw(ArgumentError("expected $(expected) params, got $(length(args))"))
    end

    params = a.last_params
    param_names = a.ansatz.param_names

    arg_idx = 1
    for i in eachindex(params)
        name = param_names[i]
        if isnan(a.hardcodes[i])
            params[i] = args[arg_idx]
            arg_idx += 1
        else
            params[i] = a.hardcodes[i]
        end
    end

    return params
end

function (a::Autoscaler)(params::Vector)
    params = get_parameters(a, params)
    return cost_function(a, params)
end
function (a::Autoscaler)(args::Vararg{<:Real})
    params = get_parameters(a, args)
    return cost_function(a, params)
end

function to_table(a::Autoscaler, args::Vararg{<:Real})
    return to_table(a, args)
end
function to_table(a::Autoscaler, args::Union{Vector,Tuple})
    params = get_parameters(a, args)
    transform!(a, params)
    if a.classic_cost_function
        master_curve = [
            standard_master_curve(a, x, size) for (x, size) in zip(a.x_val_scaled, a.sizes)
        ]
    else
        master_curve = [
            new_master_curve(a, x, size) for (x, size) in zip(a.x_val_scaled, a.sizes)
        ]
    end
    return (;
        L=copy(a.sizes),
        x=copy(a.x_val_scaled),
        y=copy(a.y_val_scaled),
        y_err=copy(a.y_err_scaled),
        master_curve,
    )
end

function cost_function(a::Autoscaler, params)
    if a.classic_cost_function
        return standard_cost_function(a, params)
    else
        return new_cost_function(a, params)
    end
end
