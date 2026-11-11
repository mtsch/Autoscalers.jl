@testset "fit_collapse and bootstrap" begin
    g(L, x) = L^-0.25 * (1 + tanh(L * (x - 0.3)) + 0.3 * L * (x - 0.3))
    exact = synthetic(g; sizes=[8, 16, 32], xs=range(0.2, 0.4; length=41), σ=2e-3)
    # inside the range all sizes cover, 0.1 / (32 / 8) = 0.025 around x_c
    window = (; eval_window=(0.28, 0.32))

    @testset "fit_collapse" begin
        s = Autoscaler(exact; window...)
        fit = fit_collapse(s, [1.1, 0.2, 0.31])
        @test fit.converged
        @test fit.params ≈ [1.0, 0.25, 0.3] atol = 3e-3
        @test fit.cost ≈ s(fit.params)
        # nothing left to fit
        s = Autoscaler(exact; window..., fix=(; a=1.0, b=0.25, x_c=0.3))
        @test fit_collapse(s, Float64[]).params == Float64[]
    end

    @testset "the model data lies on the fitted collapse" begin
        p = [1.0, 0.25, 0.3]
        for (ansatz, q) in [
            (Autoscalers.DEFAULT_ANSATZ, p),
            # affine with an offset, and non-linear in the column with errors
            (@ansatz((A, a, b, x_c) -> :L^b * ((:y ± :y_err) - A) ~ :L^a * (:x - x_c)),
             [0.05; p]),
            (@ansatz((a, b, x_c) -> :L^b * log(2 + (:y ± :y_err)) ~ :L^a * (:x - x_c)), p),
        ]
            s = Autoscaler(exact, ansatz; window...)
            model = Autoscalers.model_data(s, q)
            transform!(s, q)
            target = Autoscalers.pooled_master_curve(s)
            m = Autoscaler(s; data=model)
            transform!(m, q)
            @test maximum(abs.(m.y_val_scaled .- target) ./ m.y_err_scaled) < 1e-8
            # only the first column with errors is replaced
            @test model.x == s.data.x
        end
        # errors far below the size of the values: the scaled `y` cannot be matched to
        # `1e-10` of its error
        precise = (; exact..., y=exact.y .+ 100, y_err=fill(1e-8, length(exact.y)))
        offset = @ansatz (a, b, x_c) -> :L^b * ((:y ± :y_err) - 100) ~ :L^a * (:x - x_c)
        s = Autoscaler(precise, offset; window...)
        model = Autoscalers.model_data(s, p)
        transform!(s, p)
        target = Autoscalers.pooled_master_curve(s)
        m = Autoscaler(s; data=model)
        transform!(m, p)
        @test maximum(abs.(m.y_val_scaled .- target) ./ m.y_err_scaled) < 1e-3

        flat = Ansatz((:c,); x=:x, errors=(; y=:y_err)) do row, p
            return row.x, row.y - row.y + p.c
        end
        s = Autoscaler(exact, flat; window...)
        @test_throws ArgumentError Autoscalers.model_data(s, [1.0])
    end

    @testset "copies of an Autoscaler" begin
        s = Autoscaler(exact; window...)
        c = Autoscaler(s; fix=(; b=0.25))
        @test Autoscalers.free_parameter_names(c) == (:a, :x_c)
        @test c(1.0, 0.3) == s(1.0, 0.25, 0.3)
        @test c.x_val_scaled !== s.x_val_scaled
        shifted = Autoscaler(s; data=(; s.data..., y=s.data.y .+ 1))
        @test shifted.data.y == s.data.y .+ 1
        @test_throws ArgumentError Autoscaler(s; fix=(; c=1.0))
    end

    noisy = synthetic(g; sizes=[8, 16, 32], xs=range(0.2, 0.4; length=41), σ=2e-3,
                      rng=Xoshiro(1))
    s = Autoscaler(noisy; window...)
    estimate = fit_collapse(s, [1.1, 0.2, 0.31]).params
    n = 60
    b = bootstrap(s, estimate; n, rng=Xoshiro(2), progress=false)

    @testset "bootstrap" begin
        @test b isa BootstrapResult
        @test b.param_names == (:a, :b, :x_c)
        @test size(b.samples) == (n, 3)
        @test all(b.converged)
        @test startswith(sprint(show, b), "BootstrapResult(a, b, x_c; 60/60")
        # reproducible for a given rng, independently of threading
        serial = bootstrap(s, estimate; n, rng=Xoshiro(2), progress=false, parallel=false)
        @test serial.samples == b.samples
        @test_throws ArgumentError bootstrap(s, estimate[1:2]; n=2, progress=false)
    end

    @testset "confint and stderror" begin
        ci = confint(b)
        se = stderror(b)
        @test keys(ci) == keys(se) == (:a, :b, :x_c)
        narrow = confint(b; level=0.68)
        for (j, name) in enumerate(b.param_names)
            lo, hi = ci[name]
            @test lo < estimate[j] < hi
            @test lo ≤ narrow[name][1] < narrow[name][2] ≤ hi
            @test 0 < se[name] < hi - lo
        end
        @test_throws ArgumentError confint(b; level=1.0)
    end

    @testset "bias correction" begin
        quantiles(x, level) = (quantile(x, (1 - level) / 2), quantile(x, (1 + level) / 2))
        # with samples symmetric around the estimate, z₀ = 0: plain percentile interval
        mirrored = vcat(b.samples, 2 .* estimate' .- b.samples)
        sym = BootstrapResult(
            s, b.param_names, estimate, mirrored, trues(2n), b.method, b.options
        )
        for (j, name) in enumerate(b.param_names)
            @test all(confint(sym)[name] .≈ quantiles(mirrored[:, j], 0.95))
        end
        # samples off-centre above the estimate: the interval moves down
        shifted = b.samples .+ 0.5 .* collect(stderror(b))'
        up = BootstrapResult(
            s, b.param_names, estimate, shifted, trues(n), b.method, b.options
        )
        @test all(confint(up).a .< quantiles(shifted[:, 1], 0.95))
        # The test is dual to the interval. Checked on many off-centre samples, because
        # with few the shifted quantile levels are not resolved.
        many = estimate' .+ 0.003 .+ 0.01 .* randn(Xoshiro(4), 4000, 3)
        off = BootstrapResult(
            s, b.param_names, estimate, many, trues(4000), b.method, b.options
        )
        lo, hi = confint(off; level=0.9).a
        @test pvalue(off, :a => (lo + hi) / 2) > 0.1
        @test pvalue(off, :a => lo) ≈ 0.1 atol = 0.01
        @test pvalue(off, :a => hi) ≈ 0.1 atol = 0.01
        @test pvalue(off, :a => estimate[1] + 1) < 1e-3

        # ties count as half; samples all on one side are clamped to 1/(2n) from 0 or 1
        @test Autoscalers.bias_correction([1, 2, 2, 3], 2) == 0
        q = Autoscalers.normal_quantile
        @test Autoscalers.bias_correction([1, 2, 3, 4], 10) ≈ q(1 - 1 / 8)
        @test Autoscalers.bias_correction([1, 2, 3, 4], 0) ≈ q(1 / 8)
        # a single replicate gives a finite, zero-width interval at that replicate
        one = BootstrapResult(
            s, b.param_names, estimate, b.samples[1:1, :], trues(1), b.method, b.options
        )
        for (j, name) in enumerate(b.param_names)
            @test confint(one)[name] == (b.samples[1, j], b.samples[1, j])
        end
        # but no standard deviation
        @test_throws r"at least 2 converged" stderror(one)
    end

    @testset "pvalue" begin
        @test pvalue(b, :a => estimate[1]) > 0.3
        # beyond all samples, the p-value is the smallest one attainable on that side
        z0 = Autoscalers.bias_correction(b.samples[:, 1], estimate[1])
        Φ, Φ⁻¹ = Autoscalers.normal_cdf, Autoscalers.normal_quantile
        smallest(z) = 2Φ(Φ⁻¹(1 / (n + 1)) + z)
        @test pvalue(b, :a => 2.0) ≈ smallest(2z0)
        @test pvalue(b, :a => 0.0) ≈ smallest(-2z0)

        # resampling under the hypothesis
        kw = (; method=:null, n=20, rng=Xoshiro(3), progress=false)
        @test pvalue(b, :a => estimate[1]; kw...) > 0.05
        # forcing a far-off value pulls the curves apart: one warning, not one per refit
        far = r"bootstrap test replicates \(and the fit under the hypothesis\) extrapolate"
        @test (@test_logs (:warn, far) pvalue(b, :a => 1.3; kw...)) ≈ 1 / 21
        @test (@test_logs (:warn, far) pvalue(b, :a => 1.3, :b => 0.5; kw...)) ≈ 1 / 21

        @test_throws ArgumentError pvalue(b)
        @test_throws ArgumentError pvalue(b, :c => 1.0)
        @test_throws ArgumentError pvalue(b, :a => 1.0, :b => 0.25)
        @test_throws ArgumentError pvalue(b, :a => 1.0; method=:percentile)
        @test_throws ArgumentError pvalue(b, :a => 1.0; n=10)
    end

    @testset "no usable replicates" begin
        @test_throws ArgumentError bootstrap(s, estimate; n=0, progress=false)
        @test_throws ArgumentError pvalue(b, :a => 1.0; method=:null, n=0, progress=false)
        none = BootstrapResult(
            s, b.param_names, estimate, b.samples, falses(n), b.method, b.options
        )
        @test_throws ArgumentError confint(none)
        @test_throws ArgumentError stderror(none)
        @test_throws ArgumentError pvalue(none, :a => 1.0)
        # a constrained fit that cannot converge in one iteration
        stuck = BootstrapResult(
            s, b.param_names, estimate, b.samples, b.converged, b.method,
            Optim.Options(; iterations=1),
        )
        kw = (; method=:null, n=2, progress=false)
        @test_throws ArgumentError pvalue(stuck, :a => 1.3; kw...)
    end

    @testset "extrapolation warning" begin
        # inside the range all sizes cover: no warning
        @test_logs fit_collapse(s, estimate)
        @test !Autoscalers.extrapolates(s, estimate)
        classic = Autoscaler(noisy; classic_cost_function=true)
        @test !Autoscalers.extrapolates(classic, estimate)

        # beyond it: one warning, which suggests a window that needs no extrapolation
        wide = Autoscaler(noisy; eval_window=(0.22, 0.38))
        fit = @test_logs (:warn, r"extrapolates curves .* would need none") fit_collapse(
            wide, estimate
        )
        @test Autoscalers.extrapolates(wide, fit.params)
        _, suggested = Autoscalers.extrapolation_free_window(
            wide, Autoscalers.get_parameters(wide, fit.params)
        )
        @test 0.22 ≤ suggested[1] < 0.3 < suggested[2] ≤ 0.38
        inside = Autoscaler(noisy; eval_window=suggested)
        @test cost_breakdown(inside, fit.params).penalty == 0

        # the bootstrap and its test warn once for all refits, not once per refit
        kw = (; rng=Xoshiro(5), progress=false)
        bw = @test_logs (:warn, r"4 of 4 bootstrap fits extrapolate") bootstrap(
            wide, fit.params; n=4, kw...
        )
        @test_logs (:warn, r"of 3 bootstrap test replicates") pvalue(
            bw, :a => fit.params[1]; method=:null, n=3, kw...
        )
    end
end
