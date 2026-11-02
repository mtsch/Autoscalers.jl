"""
    lerp((x1, y1), (x2, y2), x)

Linearly interpolate at `x` between `(x1, y1)` and `(x2, y2)`.
"""
function lerp((x1, y1), (x2, y2), selected_x)
    if x1 == x2
        return y1
    else
        return (y1 * (x2 - selected_x) + y2 * (selected_x - x1)) / (x2 - x1)
    end
end

"""
    interpolate(x, y, y_err, selected_x; sq_errs=false)

Interpolate `y` and `y_err` (squared if `sq_errs=true`) with knots in `x` at point
`selected_x`.
"""
function interpolate(x, y, y_err, selected_x; sq_errs=false)
    indices = searchsorted(x, selected_x)
    if indices.start ≤ indices.stop
        if sq_errs
            return y[indices.start], y_err[indices.start]^2
        else
            return y[indices.start], y_err[indices.start]
        end
    elseif indices.start > length(x) || indices.stop ≤ 0
        # No data available
        return missing
    else
        # Indices are reversed as this is an empty range
        left = indices.stop
        right = indices.start
        x1, x2 = x[left], x[right]
        y1, y2 = y[left], y[right]
        e1, e2 = y_err[left], y_err[right]
        if sq_errs
            e1 *= e1
            e2 *= e2
        end

        return lerp((x1, y1), (x2, y2), selected_x), lerp((x1, e1), (x2, e2), selected_x)
    end
end

function all_knots(a::Autoscaler, lo, hi)
    knots = filter(x -> lo ≤ x ≤ hi, a.x_val_scaled)
    append!(knots, (lo, hi))
    unique!(knots)
    sort!(knots)
    return knots
end
function points_at!(dst, a::Autoscaler, x)
    @boundscheck if length(dst) ≠ length(a.avail_sizes)
        throw(BoundsError("bad length of `dst`"))
    end
    for (i, L) in enumerate(a.avail_sizes)
        xs, ys, y_errs = scaled_data(a, L)
        dst[i] = interpolate(xs, ys, y_errs, x; sq_errs=true)
    end
end

"""
    LerpIterator(a::Autoscaler, lo, hi)

Iterates linearly interpolated `x_val`, `y_val`, and `y_err^2` such that each unique point
in `x_val` is visited.
"""
struct LerpIterator{A<:AbstractVector{Float64}}
    x_val::Vector{A}
    y_val::Vector{A}
    y_err::Vector{A}
    lo::Float64
    hi::Float64
end

function LerpIterator(a::Autoscaler, lo, hi)
    x_val = map(L -> scaled_data(a, L).x_val, a.avail_sizes)
    y_val = map(L -> scaled_data(a, L).y_val, a.avail_sizes)
    y_err = map(L -> scaled_data(a, L).y_err, a.avail_sizes)

    return LerpIterator(x_val, y_val, y_err, lo, hi)
end

Base.IteratorSize(::LerpIterator) = Base.SizeUnknown()

"""
    lerp_points(state, x_vals, y_vals, y_errs, curr_x)

Interpolate `x_vals`, `y_vals`, and squared `y_errs` at point `curr_x` with help from the
iterator `state`.
"""
function lerp_points(indices, xs, ys, dys, curr_x)
    return map(zip(indices, xs, ys, dys)) do (i, x, y, dy)
        x_stop = x[i]
        y_stop = y[i]
        dy2_stop = dy[i]^2

        if x_stop ≠ curr_x
            i -= 1
        end
        x_start = x[i]
        y_start = y[i]
        dy2_start = dy[i]^2

        y_val = lerp((x_start, y_start), (x_stop, y_stop), curr_x)
        dy2_val = lerp((x_start, dy2_start), (x_stop, dy2_stop), curr_x)

        (y_val, dy2_val)
    end
end

"""
    should_stop(state, x_vals)

Determine whether `LerpIterator` with `state` and `x_vals` should stop.
"""
function should_stop(indices, xs)
    return any(zip(indices, xs)) do (i, x)
        i > length(x)
    end
end

"""
    min_x_val(state, x_vals)

Find the minimum `x_vals` in the `state`.
"""
function min_x_val(indices, xs)
    return minimum(zip(xs, indices)) do (x, i)
        x[i]
    end
end

"""
    next_state(state, x_vals)

Find the next `state` for a `LerpIterator` whith `x_vals`.
"""
function next_state(indices, xs, curr_x)
    for i in eachindex(indices)
        indices[i] += (xs[indices[i]] == curr_x)
    end
    return indices
end

function Base.iterate(it::LerpIterator)
    curr_x = max(it.lo, maximum(x[1] for x in it.x_val))

    state = map(it.x_val) do x
        start_index = findfirst(>(curr_x), x)
        isnothing(start_index) ? 0 : start_index
    end

    if any(iszero, state)
        return nothing
    else
        points = lerp_points(state, it.x_val, it.y_val, it.y_err, curr_x)
        return curr_x => points, state
    end
end
function Base.iterate(it::LerpIterator, state)
    if should_stop(state, it.x_val)
        return nothing
    end
    @show state

    curr_x = min_x_val(state, it.x_val)
    if curr_x > it.hi
        curr_x = it.hi
    end

    points = lerp_points(state, it.x_val, it.y_val, it.y_err, curr_x)
    if curr_x == it.hi
        state = fill(typemax(Int), length(it.x_val)) # ensure next iteration finishes
    else
        state = next_state(state, it.x_val, curr_x)
    end

    return curr_x => points, state
end


###
###
###
struct InplacePointIterator{A<:Autoscaler} <: AbstractVector{
    @NamedTuple{L::Int,in_data::Bool,x::Float64,y::Float64,dy2::Float64}
}
    scaler::A
    curr_x::Base.RefValue{Float64}
    curr_indices::Vector{Int}
    prev_x::Base.RefValue{Float64}
    prev_indices::Vector{Int}
    hi::Float64
end
function InplacePointIterator(a::Autoscaler, lo, hi)
    curr_indices = map(a.avail_sizes) do l
        x_val = scaled_data(a, l).x_val
        findfirst(≥(lo), x_val)
    end
    return InplacePointIterator(a, Ref(lo), curr_indices, Ref(-Inf), fill(0, length(curr_indices)), hi)
end

Base.size(p::InplacePointIterator) = (length(p.scaler.avail_sizes),)

function Base.getindex(p::InplacePointIterator, index)
    0 ≤ index ≤ length(p) || throw(BoundsError(p, index))
    position = p.curr_indices[index]

    x, y, dy = scaled_data(p.scaler, p.scaler.avail_sizes[index])

    x_stop = x[position]
    y_stop = y[position]
    dy2_stop = dy[position]^2
    if x_stop ≠ p.curr_x[]
        position -= 1
    end
    x_start = x[position]
    y_start = y[position]
    dy2_start = dy[position]^2

    y = lerp((x_start, y_start), (x_stop, y_stop), p.curr_x[])
    dy2 = lerp((x_start, dy2_start), (x_stop, dy2_stop), p.curr_x[])

    (; L=p.scaler.avail_sizes[index], in_data=x_start == x_stop, x=p.curr_x[], y, dy2)
end

function next!(p::InplacePointIterator)
    if p.curr_x[] ≥ p.hi
        p.curr_indices .= 0
        return false
    end
    curr_x = p.hi
    for i in eachindex(p.curr_indices)
        xs = scaled_data(p.scaler, p.scaler.avail_sizes[i]).x_val
        x = xs[p.curr_indices[i]]
        new_index = p.curr_indices[i] + (x == p.curr_x[])

        p.prev_indices .= p.curr_indices
        p.prev_x[] = p.curr_x[]
        if new_index > length(xs)
            p.curr_indices .= 0
            return false
        else
            p.curr_indices[i] = new_index
            curr_x = min(xs[new_index], curr_x)
        end
    end
    p.curr_x[] = curr_x
    return true
end

function previous(p::InplacePointIterator)
    return InplacePointIterator(p.scaler, p.prev_x, p.prev_indices, p.curr_x, p.curr_indices, p.hi)
end

function delta_squared(points::InplacePointIterator, index; skip=false)
    (; y, dy2) = points[index]

    num = den = 0.0
    for (i, p) in enumerate(points)
        skip && i == index && continue
        weight = 1 / p.dy2
        num += p.y * weight
        den += weight
    end
    Y = num / den
    return (y - Y)^2 / dy2
end
function delta_squared(points::Vector{Tuple{Float64,Float64}}, index; skip=false)
    y, dy2 = points[index]

    num = den = 0.0
    for (i, (p_y, p_dy2)) in enumerate(points)
        skip && i == index && continue
        weight = 1 / p_dy2
        num += p_y * weight
        den += weight
    end
    Y = num / den
    return (y - Y)^2 / dy2
end
