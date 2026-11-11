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

Returns a `NamedTuple` with

* `params`: the fitted free parameters;
* `cost`: the cost at `params`;
* `converged`: whether the optimiser converged;
* `result`: the Optim result of the last run, for details such as the number of
  iterations (`nothing` if all parameters are fixed).

By default this uses the Nelder–Mead simplex method with a small initial simplex around
`x0`. It is restarted `restarts` times from the previous minimum, because a single run can
stall in an elongated simplex. Any derivative-free Optim `method` and `options` can be
passed instead.

# Examples

```jldoctest
julia> data = [(; L, x, y=L^-0.25 * tanh(L * (x - 0.3)), y_err=1e-3)
               for L in (8, 16, 32) for x in 0.2:0.005:0.4];

julia> fit = fit_collapse(Autoscaler(data; eval_window=(0.28, 0.32)), [1.1, 0.2, 0.31]);

julia> fit.converged
true

julia> round.(fit.params; digits=3)  # a, b, x_c
3-element Vector{Float64}:
 0.999
 0.249
 0.3
```
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
