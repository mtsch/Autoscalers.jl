@testset "fit_collapse and bootstrap" begin
    g(L, x) = L^-0.25 * (1 + tanh(L * (x - 0.3)) + 0.3 * L * (x - 0.3))
    exact = synthetic(g; sizes=[8, 16, 32], xs=range(0.2, 0.4; length=41), σ=2e-3)
    window = (; eval_window=(0.27, 0.33))

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
            m = Autoscalers.with_data(s, model)
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
        m = Autoscalers.with_data(s, model)
        transform!(m, p)
        @test maximum(abs.(m.y_val_scaled .- target) ./ m.y_err_scaled) < 1e-3

        flat = Ansatz((:c,); x=:x, errors=(; y=:y_err)) do row, p
            return row.x, row.y - row.y + p.c
        end
        s = Autoscaler(exact, flat)
        @test_throws ArgumentError Autoscalers.model_data(s, [1.0])
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

    @testset "pvalue" begin
        # percentile: consistent with the interval, and bounded below by 2/(n + 1)
        @test pvalue(b, :a => estimate[1]) > 0.3
        @test pvalue(b, :a => 2.0) ≈ 2 / (n + 1)
        lo, hi = confint(b; level=0.9).a
        @test pvalue(b, :a => lo - 1e-9) < 0.1 + 2 / (n + 1)
        @test pvalue(b, :a => (lo + hi) / 2) > 0.1

        # resampling under the hypothesis
        kw = (; method=:null, n=20, rng=Xoshiro(3), progress=false)
        @test pvalue(b, :a => estimate[1]; kw...) > 0.05
        @test pvalue(b, :a => 1.3; kw...) ≈ 1 / 21
        @test pvalue(b, :a => 1.3, :b => 0.5; kw...) ≈ 1 / 21

        @test_throws ArgumentError pvalue(b)
        @test_throws ArgumentError pvalue(b, :c => 1.0)
        @test_throws ArgumentError pvalue(b, :a => 1.0, :b => 0.25)
        @test_throws ArgumentError pvalue(b, :a => 1.0; method=:bca)
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
end
