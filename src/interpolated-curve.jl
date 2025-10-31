struct InterpolatedCurve{A<:AbstractVector{Float64}}
    x::A
    y::A
    y_err::A
end

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




"""
    LerpIterator(x_val::Tuple, y_val::Tuple, y_err::Tuple)

Iterates linearly interpolated `x_val`, `y_val`, and `y_err^2` such that each unique point
in `x_val` is visited.
"""
struct LerpIterator{N,A<:AbstractVector{Float64}}
    x_val::NTuple{N,A}
    y_val::NTuple{N,A}
    y_err::NTuple{N,A}
    lo::Float64
    hi::Float64
end
Base.IteratorSize(::LerpIterator) = Base.SizeUnknown()

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
    lerp_points(state, x_vals, y_vals, y_errs, curr_x)

Interpolate `x_vals`, `y_vals`, and squared `y_errs` at point `curr_x` with help from the
iterator `state`.
"""
@inline lerp_points(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, _) = ()
@inline function lerp_points((i, is...), (x, xs...), (y, ys...), (dy, dys...), curr_x)
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

    return ((y_val, dy2_val), lerp_points(is, xs, ys, dys, curr_x)...)
end

"""
    should_stop(state, x_vals)

Determine whether `LerpIterator` with `state` and `x_vals` should stop.
"""
@inline should_stop(::Tuple{}, ::Tuple{}) = false
@inline function should_stop((i, is...), (x, xs...))
    return i > length(x) || should_stop(is, xs)
end

"""
    min_x_val(state, x_vals)

Find the minimum `x_vals` in the `state`.
"""
@inline min_x_val(::Tuple{}, ::Tuple{}) = Inf
@inline function min_x_val((i, is...), (x, xs...))
    return min(x[i], min_x_val(is, xs))
end

"""
    next_state(state, x_vals)

Find the next `state` for a `LerpIterator` whith `x_vals`.
"""
@inline next_state(::Tuple{}, ::Tuple{}, _) = ()
@inline function next_state((i, is...), (x, xs...), curr_x)
    if x[i] == curr_x
        i += 1
    end
    return (i, next_state(is, xs, curr_x)...)
end

function Base.iterate(it::LerpIterator{N}) where {N}
    curr_x = max(it.lo, maximum(it.x_val[i][1] for i in 1:N))

    state = ntuple(Val(N)) do index
        start_index = findfirst(>(curr_x), it.x_val[index])
        isnothing(start_index) ? 0 : start_index
    end

    if any(iszero, state)
        return nothing
    else
        points = lerp_points(state, it.x_val, it.y_val, it.y_err, curr_x)
        return curr_x => points, state
    end
end
function Base.iterate(it::LerpIterator{N}, state) where {N}
    if should_stop(state, it.x_val)
        return nothing
    end

    curr_x = min_x_val(state, it.x_val)
    if curr_x > it.hi
        curr_x = it.hi
    end

    points = lerp_points(state, it.x_val, it.y_val, it.y_err, curr_x)
    if curr_x == it.hi
        state = ntuple(Returns(typemax(Int)), Val(N)) # ensure next iteration finishes
    else
        state = next_state(state, it.x_val, curr_x)
    end

    return curr_x => points, state
end
