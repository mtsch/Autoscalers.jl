# Classic cost function of Houdayer & Hartmann, Phys. Rev. B 70, 014418 (2004),
# https://doi.org/10.1103/PhysRevB.70.014418, Appendix A: every point is compared to a
# weighted linear fit through the neighbouring points of all other curves (sizes in the
# paper).

"""
    bracket(x, selected_x)

The index ``j`` of the two neighbouring points ``j`` and ``j + 1`` of a curve with
``x_j ≤ x_\\mathrm{sel} ≤ x_{j+1}``, where ``x_\\mathrm{sel}`` is `selected_x`, or
`nothing` if the curve does not bracket it
([Houdayer & Hartmann (2004)](https://doi.org/10.1103/PhysRevB.70.014418), Appendix A).
"""
function bracket(x, selected_x)
    n = length(x)
    j = searchsortedlast(x, selected_x)
    # On an interior knot both adjacent pairs bracket `selected_x`; the pair starting at the
    # knot matches the reference implementation (autoScale). On the last knot only the pair
    # ending there does.
    if j == n && x[n] == selected_x
        return n - 1
    elseif j == 0 || j == n
        return nothing
    end
    return j
end

"""
    standard_master_curve(c::Autoscaler, selected_x, selected_curve)

Weighted linear fit through the neighbours of `selected_x` in all other curves (see
[`bracket`](@ref)), evaluated at `selected_x` (Eqs. A2 and A3). Returns `(Y, dY2)` or
`missing` if no other curve brackets `selected_x`.
"""
function standard_master_curve(c::Autoscaler, selected_x, selected_curve)
    K = Kx = Ky = Kxx = Kxy = 0.0
    for curve in 1:n_curves(c)
        curve == selected_curve && continue
        x, y, y_err = scaled_data(c, curve)
        j = bracket(x, selected_x)
        isnothing(j) && continue
        for k in (j, j + 1)
            w = 1 / y_err[k]^2
            K += w
            Kx += w * x[k]
            Ky += w * y[k]
            Kxx += w * x[k]^2
            Kxy += w * x[k] * y[k]
        end
    end
    K == 0 && return missing
    Δ = K * Kxx - Kx^2

    Y = (Kxx * Ky - Kx * Kxy) / Δ + selected_x * (K * Kxy - Kx * Ky) / Δ
    dY2 = (Kxx - 2selected_x * Kx + selected_x^2 * K) / Δ

    return Y, dY2
end

"""
    standard_cost_function(c::Autoscaler, params)

Quality ``S`` of Eq. A1: the mean of ``(y - Y)^2 / (δy^2 + δY^2)`` over all points for
which the master curve ``Y ± δY`` is defined. `Inf` if it is defined for none.

# References

- J. Houdayer and A. K. Hartmann, *Low-temperature behavior of two-dimensional Gaussian
  Ising spin glasses*, Phys. Rev. B **70**, 014418 (2004),
  [doi:10.1103/PhysRevB.70.014418](https://doi.org/10.1103/PhysRevB.70.014418).
"""
function standard_cost_function(c::Autoscaler, params)
    transform!(c, params)
    s = 0.0
    n = 0
    for i in eachindex(c.x_val_scaled)
        master = standard_master_curve(c, c.x_val_scaled[i], c.curve[i])
        if !ismissing(master)
            Y, dY2 = master
            s += (c.y_val_scaled[i] - Y)^2 / (c.y_err_scaled[i]^2 + dY2)
            n += 1
        end
    end
    return n == 0 ? Inf : s / n
end
