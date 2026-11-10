# Coverage of bootstrap percentile intervals and size of the bootstrap tests on synthetic
# data with known parameters. Too slow for CI; run with e.g.
#
#     julia --project -t auto test/scripts/bootstrap_coverage.jl [datasets] [replicates]
#
# A well-calibrated interval of level 1 - α contains the true value in a fraction 1 - α of
# the datasets, and a test of the true value rejects at level α in a fraction α of them.
using Autoscalers
using Printf
using Random

const TRUTH = (; a=1.0, b=0.25, x_c=0.3)

g(L, x) = L^-TRUTH.b * (1 + tanh(L^TRUTH.a * (x - TRUTH.x_c)) + 0.3 * L * (x - TRUTH.x_c))

function dataset(rng; sizes=[8, 16, 32], xs=range(0.2, 0.4; length=41), σ=2e-3)
    L = repeat(sizes; inner=length(xs))
    x = repeat(xs, length(sizes))
    y = g.(L, x) .+ σ .* randn(rng, length(x))
    return (; L, x, y, y_err=fill(σ, length(x)))
end

# Inside the range all sizes cover at the truth, 0.1 / (32 / 8)^a = 0.025 around x_c, so
# that no curve has to be extrapolated; beyond it, linear extrapolation biases the fit.
const WINDOW = (0.28, 0.32)

# Binomial standard error of an observed fraction, to judge the deviations.
binomial_se(p, n) = sqrt(p * (1 - p) / n)

function main(n_datasets, n_replicates; n_null=max(1, n_datasets ÷ 4))
    rng = Xoshiro(20261110)
    names = keys(TRUTH)
    levels = (0.68, 0.95)
    methods = (:percentile, :bc)
    covered = Dict((m, name, level) => 0 for m in methods, name in names, level in levels)
    rejected = Dict((m, name) => 0 for m in methods, name in names)
    rejected_null = 0
    for k in 1:n_datasets
        s = Autoscaler(dataset(rng); eval_window=WINDOW)
        estimate = fit_collapse(s, [1.1, 0.2, 0.31]).params
        b = bootstrap(s, estimate; n=n_replicates, rng, progress=false)
        for method in methods, level in levels
            intervals = confint(b; level, method)
            for name in names
                lo, hi = intervals[name]
                covered[(method, name, level)] += lo ≤ TRUTH[name] ≤ hi
            end
        end
        for method in methods, name in names
            rejected[(method, name)] += pvalue(b, name => TRUTH[name]; method) < 0.05
        end
        if k ≤ n_null
            p = pvalue(b, :a => TRUTH.a; method=:null, n=n_replicates, rng, progress=false)
            rejected_null += p < 0.05
        end
        print("\rdataset $k/$n_datasets")
    end
    println()
    for method in methods
        for level in levels
            se = binomial_se(level, n_datasets)
            @printf("%-10s %.0f%% intervals (±%.3f):", method, 100level, se)
            for name in names
                @printf("  %s %.3f", name, covered[(method, name, level)] / n_datasets)
            end
            println()
        end
        se = binomial_se(0.05, n_datasets)
        @printf("%-10s test, rejections at 5%% (±%.3f):", method, se)
        for name in names
            @printf("  %s %.3f", name, rejected[(method, name)] / n_datasets)
        end
        println()
    end
    @printf(
        "null-resampling test of a, rejections at 5%% (±%.3f): %.3f over %d datasets\n",
        binomial_se(0.05, n_null), rejected_null / n_null, n_null,
    )
end

main(parse(Int, get(ARGS, 1, "200")), parse(Int, get(ARGS, 2, "400")))
