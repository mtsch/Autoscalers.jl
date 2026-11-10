# A small relative initial simplex: Optim's default moves each parameter by 50 %, which
# can push the scaled sizes apart until a cost is Inf, and Nelder-Mead then stops before
# its first iteration.
default_fit_method() = NelderMead(; initial_simplex=Optim.AffineSimplexer(0.005, 0.02))
default_fit_options() = Optim.Options(; iterations=10_000, g_tol=1e-8)

"""
    fit_collapse(
        a::Autoscaler, x0;
        method=NelderMead(...), options=Optim.Options(...), restarts=1,
    )

Minimise the cost of `a` over its free parameters with Optim.jl, starting from `x0`.
Returns a `NamedTuple` `(; params, cost, converged, result)`, where `result` is the last
Optim result.

By default this uses the Nelder–Mead simplex method (Nelder & Mead, 1965), with a small
initial simplex around `x0`, and restarts it `restarts` times from the previous minimum,
because a single run can stall in an elongated simplex. Any derivative-free Optim
`method` and `options` can be passed instead.

# References

- J. A. Nelder and R. Mead, *A simplex method for function minimization*, Comput. J.
  **7**, 308 (1965), [doi:10.1093/comjnl/7.4.308](https://doi.org/10.1093/comjnl/7.4.308).
"""
function fit_collapse(
    a::Autoscaler, x0;
    method=default_fit_method(), options=default_fit_options(), restarts=1,
)
    x0 = Float64.(collect(x0))
    if isempty(x0)
        return (; params=x0, cost=a(x0), converged=true, result=nothing)
    end
    result = optimize(a, x0, method, options)
    for _ in 1:restarts
        result = optimize(a, Optim.minimizer(result), method, options)
    end
    return (;
        params=Optim.minimizer(result),
        cost=Optim.minimum(result),
        converged=Optim.converged(result),
        result,
    )
end
