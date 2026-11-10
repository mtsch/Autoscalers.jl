const REFERENCES_BOOTSTRAP = """
- B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall
  (1993), [doi:10.1201/9780429246593](https://doi.org/10.1201/9780429246593) (CRC
  reprint).
- A. C. Davison and D. V. Hinkley, *Bootstrap Methods and their Application*, Cambridge
  University Press (1997),
  [doi:10.1017/CBO9780511802843](https://doi.org/10.1017/CBO9780511802843).
"""

const REFERENCE_HALL_WILSON = """
- P. Hall and S. R. Wilson, *Two guidelines for bootstrap hypothesis testing*, Biometrics
  **47**, 757 (1991), [doi:10.2307/2532163](https://doi.org/10.2307/2532163).
"""

"""
    with_data(a::Autoscaler, data)

A copy of `a` that uses `data`, which must have the same columns and rows as `a.data`,
and has its own buffers, so it can be evaluated concurrently with `a`.
"""
function with_data(a::Autoscaler, data)
    n = length(a.sizes)
    return Autoscaler(
        a.ansatz, data, a.errors,
        zeros(n), zeros(n), zeros(n),
        a.sizes, a.avail_sizes, a.size_map,
        Float64[], Float64[], Float64[],
        copy(a.hardcodes), zeros(length(a.last_params)),
        a.classic_cost_function, a.use_all_sizes, a.window_width,
        a.eval_window, a.eval_mask, a.penalty,
    )
end

"""
    with_fixed(a::Autoscaler, fixed)

A copy of `a` with the parameters in `fixed` (pairs `name => value`) hardcoded as well.
"""
function with_fixed(a::Autoscaler, fixed)
    result = with_data(a, a.data)
    for (name, value) in fixed
        result.hardcodes[findfirst(==(name), a.ansatz.param_names)] = value
    end
    return result
end

"""
    pooled_master_curve(a::Autoscaler)

The master curve at the scaled `x` of every data point: the error-weighted mean of all
sizes that cover it, including the point's own size, so it is defined at every point.
Uses the current scaling of `a`.
"""
function pooled_master_curve(a::Autoscaler)
    return map(a.x_val_scaled) do x
        numerator = denominator = 0.0
        for size in a.avail_sizes
            xs, ys, dys = scaled_data(a, size)
            point = interpolate(xs, ys, dys, x; sq_errs=true)
            if !ismissing(point)
                numerator += point[1] / point[2]
                denominator += 1 / point[2]
            end
        end
        return numerator / denominator
    end
end

struct ModelInversion end

"""
    scaled_y_and_slope(a::Autoscaler, i, p, column, value)

Scaled `y` of data point `i` at parameters `p` with `column` set to `value`, and its
derivative with respect to that value.
"""
function scaled_y_and_slope(a::Autoscaler, i, p, column, value)
    tag = ForwardDiff.Tag(ModelInversion(), typeof(value))
    names = keys(a.data)
    row = NamedTuple{names}(map(names) do c
        if c === column
            return Dual{typeof(tag)}(value, one(value))
        else
            return a.data[c][i]
        end
    end)
    _, y = a.ansatz.f(row, p)
    if y isa Dual
        return ForwardDiff.value(y), ForwardDiff.partials(y, 1)
    else
        return Float64(y), 0.0
    end
end

"""
    model_data(a::Autoscaler, params; maxiter=50)

Data of the fitted collapse at the free parameters `params`: the first column with errors
is replaced by the values whose scaled `y` lies exactly on the `pooled_master_curve`.
The ansatz is inverted point by point with Newton's method, which takes one step when the
scaled `y` is affine in that column.
"""
function model_data(a::Autoscaler, params; maxiter=50)
    full = get_parameters(a, params)
    transform!(a, full)
    target = pooled_master_curve(a)
    column = first(keys(a.ansatz.errors))
    p = parameter_tuple(a.ansatz, full)
    values = copy(a.data[column])
    for i in eachindex(values)
        tolerance = 1e-10 * a.y_err_scaled[i]
        converged = false
        for _ in 1:maxiter
            y, slope = scaled_y_and_slope(a, i, p, column, values[i])
            residual = target[i] - y
            if abs(residual) ≤ tolerance
                converged = true
                break
            end
            if iszero(slope)
                throw(ArgumentError(
                    "the scaled `y` does not depend on `$column` at data point $i, so " *
                    "the fitted collapse cannot be mapped back to the data"
                ))
            end
            values[i] += residual / slope
        end
        if !converged
            throw(ArgumentError(
                "could not invert the ansatz for `$column` at data point $i in $maxiter " *
                "Newton steps"
            ))
        end
    end
    return merge(a.data, NamedTuple{(column,)}((values,)))
end

"""
    synthetic_data(rng, model, a::Autoscaler)

`model` with Gaussian noise of the measured errors added to every column with errors.
"""
function synthetic_data(rng, model, a::Autoscaler)
    return map(model, a.errors) do values, errors
        if isnothing(errors)
            return values
        else
            return values .+ last(errors) .* randn(rng, length(values))
        end
    end
end

"""
    replicate!(f, n; rng, parallel, progress, desc)

Call `f(i, rng_i)` for `i in 1:n`, each with its own random number generator seeded from
`rng`, so the results do not depend on how the calls are scheduled over threads.
"""
function replicate!(f, n; rng, parallel, progress, desc)
    seeds = rand(rng, UInt64, n)
    meter = Progress(n; desc, enabled=progress)
    if parallel
        Threads.@threads for i in 1:n
            f(i, Xoshiro(seeds[i]))
            next!(meter)
        end
    else
        for i in 1:n
            f(i, Xoshiro(seeds[i]))
            next!(meter)
        end
    end
    finish!(meter)
    return nothing
end

"""
    BootstrapResult

Result of [`bootstrap`](@ref). Use [`confint`](@ref), [`stderror`](@ref) and
[`pvalue`](@ref) on it. Only the replicates whose fit converged are used.

* `estimate`: the free parameters the bootstrap was run at.
* `samples`: one row of refitted free parameters per replicate.
* `converged`: whether each replicate's fit converged.
"""
struct BootstrapResult{A<:Autoscaler,M,O}
    autoscaler::A
    param_names::Tuple{Vararg{Symbol}}
    estimate::Vector{Float64}
    samples::Matrix{Float64}
    converged::BitVector
    method::M
    options::O
end

function converged_samples(b::BootstrapResult)
    if !any(b.converged)
        throw(ArgumentError("no bootstrap replicate converged, so there is nothing to use"))
    end
    return b.samples[b.converged, :]
end

function Base.show(io::IO, b::BootstrapResult)
    n = length(b.converged)
    print(io, "BootstrapResult(", join(b.param_names, ", "), "; ", count(b.converged), "/")
    print(io, n, " replicates converged)")
end

"""
    bootstrap(
        a::Autoscaler, estimate;
        n=1000, rng=Random.default_rng(), parallel=true, progress=true,
        method=NelderMead(...), options=Optim.Options(...),
    )

Parametric bootstrap of the free parameters around their fitted values `estimate`
(Efron & Tibshirani, 1993; Davison & Hinkley, 1997).

The fitted collapse is taken as the true model: at `estimate`, the master curve of all
sizes (the error-weighted mean of the sizes covering each point) is mapped back to the
data by inverting the ansatz for the first column with errors. Each of the `n`
replicates adds Gaussian noise of the measured errors to that model (and to the other
columns with errors), and refits the parameters with [`fit_collapse`](@ref), using
`method` and `options`, starting from `estimate`.

Replicates run on all Julia threads if `parallel` (start Julia with `--threads`), and
are reproducible for a given `rng` regardless of the number of threads. `progress` shows
a progress bar.

The spread of the samples estimates the statistical error only. Systematic errors, for
example from the choice of `eval_window` or corrections to scaling, are not included.

# References

$REFERENCES_BOOTSTRAP
"""
function bootstrap(
    a::Autoscaler, estimate;
    n=1000, rng=Random.default_rng(), parallel=true, progress=true,
    method=default_fit_method(), options=default_fit_options(),
)
    n ≥ 1 || throw(ArgumentError("`n` must be positive"))
    names = free_parameter_names(a)
    estimate = Float64.(collect(estimate))
    if length(estimate) ≠ length(names)
        throw(ArgumentError("expected $(length(names)) parameters $names"))
    end
    model = model_data(a, estimate)
    samples = Matrix{Float64}(undef, n, length(names))
    # Not a BitVector: its bits share words, so concurrent writes from threads would race.
    converged = fill(false, n)
    replicate!(n; rng, parallel, progress, desc="Bootstrap: ") do i, rng_i
        s = with_data(a, synthetic_data(rng_i, model, a))
        result = fit_collapse(s, estimate; method, options)
        samples[i, :] .= result.params
        converged[i] = result.converged
    end
    n_failed = n - count(converged)
    if n_failed > 0
        @warn "$n_failed of $n bootstrap fits did not converge and are left out."
    end
    converged = BitVector(converged)
    return BootstrapResult(a, names, estimate, samples, converged, method, options)
end

"""
    confint(b::BootstrapResult; level=0.95)

Bootstrap percentile confidence intervals (Efron & Tibshirani, 1993) of every free
parameter, as a `NamedTuple` of `(lower, upper)`. Percentile intervals follow the shape
of the bootstrap distribution, so they can be asymmetric around the estimate.

# References

$REFERENCES_BOOTSTRAP
"""
function StatsAPI.confint(b::BootstrapResult; level=0.95)
    0 < level < 1 || throw(ArgumentError("`level` must be in (0, 1)"))
    α = 1 - level
    intervals = map(eachcol(converged_samples(b))) do column
        return (quantile(column, α / 2), quantile(column, 1 - α / 2))
    end
    return NamedTuple{b.param_names}(Tuple(intervals))
end

"""
    stderror(b::BootstrapResult)

Bootstrap standard errors (Efron & Tibshirani, 1993) of every free parameter, as a
`NamedTuple`: the standard deviation of the samples. For asymmetric errors, prefer
[`confint`](@ref).

# References

$REFERENCES_BOOTSTRAP
"""
function StatsAPI.stderror(b::BootstrapResult)
    return NamedTuple{b.param_names}(Tuple(std.(eachcol(converged_samples(b)))))
end

"""
    pvalue(b::BootstrapResult, name => value...; method=:percentile, kwargs...)

Two-sided p-value of the hypothesis that the parameters `name` equal `value`.

* `method=:percentile` (default, no extra fits) tests one parameter against its
  bootstrap distribution: `p = 2 min(P(θ* ≤ value), P(θ* ≥ value))`, with the `+1`
  correction of Davison & Hinkley (1997) so that `p ≥ 2/(n + 1)`. It is the test dual to
  the percentile interval of [`confint`](@ref): `p < 1 - level` exactly when `value` lies
  outside the interval of that `level`.
* `method=:null` follows the first guideline of Hall & Wilson (1991) and resamples under
  the hypothesis. The parameters are refitted with `name` fixed at `value`; replicates
  are drawn from that constrained collapse, and each is fitted with and without the
  constraint. The p-value is the fraction of replicates whose cost increase from the
  constraint is at least the observed one (again with the `+1` correction). It accepts
  several parameters at once and needs `2n` fits. The keyword arguments `n` (default:
  the number of replicates of `b`), `rng`, `parallel` and `progress` are as for
  [`bootstrap`](@ref).

# References

$REFERENCES_BOOTSTRAP$REFERENCE_HALL_WILSON
"""
function StatsAPI.pvalue(
    b::BootstrapResult, hypotheses::Pair{Symbol,<:Real}...; method=:percentile, kwargs...
)
    isempty(hypotheses) && throw(ArgumentError("no hypothesis given"))
    for (name, _) in hypotheses
        if name ∉ b.param_names
            throw(ArgumentError(
                "`$name` is not a free parameter, expected one of $(b.param_names)"
            ))
        end
    end
    if method == :percentile
        isempty(kwargs) || throw(ArgumentError("`method=:percentile` takes no options"))
        if length(hypotheses) ≠ 1
            throw(ArgumentError(
                "`method=:percentile` tests one parameter; use `method=:null` for several"
            ))
        end
        return percentile_pvalue(b, only(hypotheses)...)
    elseif method == :null
        return null_pvalue(b, hypotheses; kwargs...)
    else
        throw(ArgumentError("unknown `method` $method, expected `:percentile` or `:null`"))
    end
end

function percentile_pvalue(b::BootstrapResult, name, value)
    samples = converged_samples(b)[:, findfirst(==(name), b.param_names)]
    n = length(samples)
    below = (1 + count(≤(value), samples)) / (n + 1)
    above = (1 + count(≥(value), samples)) / (n + 1)
    return min(1.0, 2 * min(below, above))
end

function null_pvalue(
    b::BootstrapResult, hypotheses;
    n=length(b.converged), rng=Random.default_rng(), parallel=true, progress=true,
)
    n ≥ 1 || throw(ArgumentError("`n` must be positive"))
    a = b.autoscaler
    constrained = with_fixed(a, hypotheses)
    fixed_names = first.(hypotheses)
    free = [i for (i, name) in enumerate(b.param_names) if name ∉ fixed_names]
    fit = fit_collapse(constrained, b.estimate[free]; b.method, b.options)
    if !fit.converged
        throw(ArgumentError(
            "the fit under the hypothesis did not converge; adjust `options` of `bootstrap`"
        ))
    end
    observed = fit.cost - a(b.estimate)

    # Start the unconstrained fits from the constrained solution, which lies on the null.
    start = copy(b.estimate)
    start[free] .= fit.params
    for (name, value) in hypotheses
        start[findfirst(==(name), b.param_names)] = value
    end
    model = model_data(constrained, fit.params)
    statistics = fill(NaN, n)
    replicate!(n; rng, parallel, progress, desc="Bootstrap test: ") do i, rng_i
        data = synthetic_data(rng_i, model, a)
        free_fit = fit_collapse(with_data(a, data), start; b.method, b.options)
        null_data = with_data(constrained, data)
        null_fit = fit_collapse(null_data, fit.params; b.method, b.options)
        if free_fit.converged && null_fit.converged
            statistics[i] = null_fit.cost - free_fit.cost
        end
    end
    statistics = filter(!isnan, statistics)
    n_failed = n - length(statistics)
    if isempty(statistics)
        throw(ArgumentError("no bootstrap test replicate converged"))
    elseif n_failed > 0
        @warn "$n_failed of $n bootstrap test replicates did not converge and are left out."
    end
    return (1 + count(≥(observed), statistics)) / (length(statistics) + 1)
end
