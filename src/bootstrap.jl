"""
    pooled_master_curve(a::Autoscaler)

The master curve at the scaled `x` of every data point: the error-weighted mean of all
curves that cover it, including the point's own curve, so it is defined at every point.
Uses the current scaling of `a`.
"""
function pooled_master_curve(a::Autoscaler)
    return map(a.x_val_scaled) do x
        numerator = denominator = 0.0
        for c in 1:n_curves(a)
            xs, ys, dys = scaled_data(a, c)
            point = interpolate(xs, ys, dys, x; sq_errs=true)
            if !ismissing(point)
                numerator += point[1] / point[2]
                denominator += 1 / point[2]
            end
        end
        return numerator / denominator
    end
end

# Tags the duals that differentiate the ansatz with respect to the column being solved
# for in `model_data`, so they cannot be confused with the error-propagation duals.
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

Data that lies exactly on the fitted collapse at the free parameters `params`. For each
data point, the value in the first column with errors is replaced by the value whose
scaled `y` equals the `pooled_master_curve` at that point. The ansatz is solved for it with
Newton's method, which needs a single step when the scaled `y` is linear in that column.
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
            step = residual / slope
            values[i] += step
            # With errors far below the size of the values, `tolerance` can be finer than
            # the rounding of the scaled `y`; then a step within rounding is converged.
            if abs(step) ≤ 4eps(abs(values[i]))
                converged = true
                break
            end
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
        n=1000, rng=Random.default_rng(), parallel=true, progress=isinteractive(),
        method=NelderMead(...), options=Optim.Options(...),
    )

Parametric bootstrap of the free parameters around their fitted values `estimate`
(Efron & Tibshirani, 1993; Davison & Hinkley, 1997).

The fitted collapse is taken as the true model: at `estimate`, the master curve of all
curves (the error-weighted mean of the curves covering each point) is mapped back to the
data by inverting the ansatz for the first column with errors. Each of the `n`
replicates adds Gaussian noise of the measured errors to that model (and to the other
columns with errors), and refits the parameters with [`fit_collapse`](@ref), using
`method` and `options`, starting from `estimate`.

Replicates run on all Julia threads if `parallel` (start Julia with `--threads`), and
are reproducible for a given `rng` regardless of the number of threads. `progress` shows
a progress bar (by default in interactive sessions).

The spread of the samples estimates the statistical error only. Systematic errors, for
example from the choice of `eval_window` or corrections to scaling, are not included.

# References

- B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall
  (1993), [doi:10.1201/9780429246593](https://doi.org/10.1201/9780429246593) (CRC
  reprint).
- A. C. Davison and D. V. Hinkley, *Bootstrap Methods and their Application*, Cambridge
  University Press (1997),
  [doi:10.1017/CBO9780511802843](https://doi.org/10.1017/CBO9780511802843).
"""
function bootstrap(
    a::Autoscaler, estimate;
    n=1000, rng=Random.default_rng(), parallel=true, progress=isinteractive(),
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
        s = Autoscaler(a; data=synthetic_data(rng_i, model, a))
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

normal_cdf(z) = erfc(-z / √2) / 2
normal_quantile(p) = -√2 * erfcinv(2p)

"""
    bias_correction(samples, estimate)

The bias correction ``z_0 = Φ^{-1}(P(θ^* < \\hat{θ}))`` of the BC interval (Efron, 1987),
counting ties as half. The fraction is kept within ``1/(2n)`` of 0 and 1, so that ``z_0``
stays finite.
"""
function bias_correction(samples, estimate)
    n = length(samples)
    below = count(<(estimate), samples) + count(==(estimate), samples) / 2
    return normal_quantile(clamp(below / n, 1 / 2n, 1 - 1 / 2n))
end

"""
    confint(b::BootstrapResult; level=0.95)

Bias-corrected percentile (BC) confidence intervals (Efron, 1987) of every free
parameter, as a `NamedTuple` of `(lower, upper)`.

The bounds are the quantiles of the bootstrap samples at the levels
``Φ(2z_0 ± z_{α/2})``, where ``Φ`` is the standard normal distribution function,
`level` ``= 1 - α``, and ``z_0 = Φ^{-1}(P(θ^* < \\hat{θ}))`` measures how far the
bootstrap distribution is off-centre from the estimate. For a centred distribution
(``z_0 = 0``) this is the plain percentile interval. The intervals follow the shape of the
bootstrap distribution, so they can be asymmetric around the estimate, and need no extra
fits. On synthetic collapses they are close to their nominal coverage (see
`test/scripts/bootstrap_coverage.jl`). A large ``|z_0|`` moves the levels towards 0 or 1,
so such quantiles need correspondingly many replicates (``n \\gtrsim 1000``).

Other intervals can be computed from the samples in `b.samples` (rows of replicates whose
fit converged are marked in `b.converged`).

# References

- B. Efron, *Better bootstrap confidence intervals*, J. Am. Stat. Assoc. **82**, 171
  (1987),
  [doi:10.1080/01621459.1987.10478410](https://doi.org/10.1080/01621459.1987.10478410).
"""
function StatsAPI.confint(b::BootstrapResult; level=0.95)
    0 < level < 1 || throw(ArgumentError("`level` must be in (0, 1)"))
    z = normal_quantile((1 - level) / 2)
    intervals = map(eachcol(converged_samples(b)), b.estimate) do samples, estimate
        z0 = bias_correction(samples, estimate)
        lower = quantile(samples, normal_cdf(2z0 + z))
        upper = quantile(samples, normal_cdf(2z0 - z))
        return (lower, upper)
    end
    return NamedTuple{b.param_names}(Tuple(intervals))
end

"""
    stderror(b::BootstrapResult)

Bootstrap standard errors of every free parameter, as a `NamedTuple`: the standard
deviation of the samples. For asymmetric errors, prefer [`confint`](@ref).
"""
function StatsAPI.stderror(b::BootstrapResult)
    return NamedTuple{b.param_names}(Tuple(std.(eachcol(converged_samples(b)))))
end

"""
    pvalue(b::BootstrapResult, name => value...; method=:bc, kwargs...)

Two-sided p-value of the hypothesis that the parameters `name` equal `value`.

* `method=:bc` (default, no extra fits) tests one parameter and is the test dual to the
  BC interval of [`confint`](@ref) (Efron, 1987): ``p < α`` exactly when `value` lies
  outside the interval of `level` ``= 1 - α``, up to the resolution of the replicates.
  The fractions ``P`` of samples below and above `value`, with the ``+1`` correction
  (Davison & Hinkley, 1997, see [`bootstrap`](@ref)), are mapped through
  ``Φ(Φ^{-1}(P) ∓ 2z_0)``, and ``p`` is twice the smaller one.
  On the side towards which the BC interval is stretched, the smallest attainable
  p-value is ``2Φ(Φ^{-1}(1/(n + 1)) + 2|z_0|)``: about 0.005 for ``n = 1000`` and
  ``|z_0| = 0.15``, but 0.06 for ``|z_0| = 0.6``. To reject strongly off-centre cases,
  use more replicates or `method=:null`.
* `method=:null` follows the first guideline of Hall & Wilson (1991) and resamples under
  the hypothesis. The parameters are refitted with `name` fixed at `value`; replicates
  are drawn from that constrained collapse, and each is fitted with and without the
  constraint. The p-value is the fraction of replicates whose cost increase from the
  constraint is at least the observed one (again with the ``+1`` correction). It accepts
  several parameters at once and needs `2n` fits. The keyword arguments `n` (default:
  the number of replicates of `b`), `rng`, `parallel` and `progress` are as for
  [`bootstrap`](@ref).

# References

- B. Efron, *Better bootstrap confidence intervals*, J. Am. Stat. Assoc. **82**, 171
  (1987),
  [doi:10.1080/01621459.1987.10478410](https://doi.org/10.1080/01621459.1987.10478410).
- P. Hall and S. R. Wilson, *Two guidelines for bootstrap hypothesis testing*, Biometrics
  **47**, 757 (1991), [doi:10.2307/2532163](https://doi.org/10.2307/2532163).
"""
function StatsAPI.pvalue(
    b::BootstrapResult, hypotheses::Pair{Symbol,<:Real}...; method=:bc, kwargs...
)
    isempty(hypotheses) && throw(ArgumentError("no hypothesis given"))
    for (name, _) in hypotheses
        if name ∉ b.param_names
            throw(ArgumentError(
                "`$name` is not a free parameter, expected one of $(b.param_names)"
            ))
        end
    end
    if method == :bc
        isempty(kwargs) || throw(ArgumentError("`method=:bc` takes no options"))
        if length(hypotheses) ≠ 1
            throw(ArgumentError(
                "`method=:bc` tests one parameter; use `method=:null` for several"
            ))
        end
        return bc_pvalue(b, only(hypotheses)...)
    elseif method == :null
        return null_pvalue(b, hypotheses; kwargs...)
    else
        throw(ArgumentError("unknown `method` $method, expected `:bc` or `:null`"))
    end
end

function bc_pvalue(b::BootstrapResult, name, value)
    j = findfirst(==(name), b.param_names)
    samples = converged_samples(b)[:, j]
    n = length(samples)
    below = (1 + count(≤(value), samples)) / (n + 1)
    above = (1 + count(≥(value), samples)) / (n + 1)
    # `value` is the lower BC bound at level 1 - p when Φ(2z₀ + z_{p/2}) = P(θ* ≤ value),
    # and similarly for the upper bound.
    z0 = bias_correction(samples, b.estimate[j])
    below = normal_cdf(normal_quantile(below) - 2z0)
    above = normal_cdf(normal_quantile(above) + 2z0)
    return min(1.0, 2 * min(below, above))
end

function null_pvalue(
    b::BootstrapResult, hypotheses;
    n=length(b.converged), rng=Random.default_rng(), parallel=true,
    progress=isinteractive(),
)
    n ≥ 1 || throw(ArgumentError("`n` must be positive"))
    a = b.autoscaler
    constrained = Autoscaler(a; fix=(; hypotheses...))
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
        free_fit = fit_collapse(Autoscaler(a; data), start; b.method, b.options)
        null_data = Autoscaler(constrained; data)
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
