# Regression test against the example of autoScale (https://github.com/omelchert/autoScale),
# the reference implementation of the classic cost function. See data/autoscale/README.md.

function read_autoscale_data(dir)
    L = Int[]
    x = Float64[]
    y = Float64[]
    y_err = Float64[]
    for l in (128, 256, 512), line in eachline(joinpath(dir, "orderParam_L$l.dat"))
        startswith(line, "#") && continue
        xi, yi, ei = parse.(Float64, split(line))
        push!(L, l)
        push!(x, xi)
        push!(y, yi)
        push!(y_err, ei)
    end
    return (; L, x, y, y_err)
end

"""
    classic_cost_xr(s, params, lo, hi)

Classic cost scoring only points with scaled `x` in `[lo, hi]`, like autoScale's `-xr`
option. Points outside the range still serve as neighbours.
"""
function classic_cost_xr(s, params, lo, hi)
    transform!(s, params)
    total = 0.0
    n = 0
    for i in eachindex(s.x_val_scaled)
        lo ≤ s.x_val_scaled[i] ≤ hi || continue
        m = standard_master_curve(s, s.x_val_scaled[i], s.curve[i])
        ismissing(m) && continue
        total += (s.y_val_scaled[i] - m[1])^2 / (s.y_err_scaled[i]^2 + m[2])
        n += 1
    end
    return n == 0 ? Inf : total / n
end

# Optim's default initial simplex moves x_c by 50 %, where the sizes no longer overlap and the
# classic cost is Inf, which stops Nelder-Mead before its first iteration.
function fit_local(f, x0)
    method = NelderMead(; initial_simplex=Optim.AffineSimplexer(0.0, 0.02))
    opts = Optim.Options(; iterations=10_000, g_tol=1e-8)
    res = optimize(f, x0, method, opts)
    res = optimize(f, Optim.minimizer(res), method, opts)
    return Optim.minimizer(res)
end

@testset "autoScale reference data" begin
    data = read_autoscale_data(joinpath(@__DIR__, "data", "autoscale"))
    @test length(data.x) == 131
    classic = Autoscaler(data; classic_cost_function=true)

    # S printed for `-xr -0.75 0.75`. Within the rounding of the printed parameters, S varies
    # by 2e-4 here; other ranges are near jumps of the classic cost and vary by percents.
    p = [0.746708, 0.104098, 0.592709]
    @test classic_cost_xr(classic, p, -0.75, 0.75) ≈ 0.769959 rtol = 1e-3

    # error.out: best fit for `-xr -1 1` with its S+1 error bars (the larger side)
    p_ref = [0.748386, 0.104841, 0.592686]
    p_err = [0.007462, 0.000975, 0.000075]
    # windows matching `-xr -1 1`: the raw x range that the largest size scales onto [-1, 1],
    # and a window width giving an overlap window of about [-0.9, 0.9]
    w = 512^-p_ref[1]
    fixed = Autoscaler(data; eval_window=(p_ref[3] - w, p_ref[3] + w))
    integral = Autoscaler(data; window_width=0.3)

    # all three cost functions must find autoScale's best fit within its error bars
    x0 = [0.75, 0.104, 0.5927]
    @testset "$name" for (name, cost) in [
        "classic, -xr -1 1" => p -> classic_cost_xr(classic, p, -1, 1),
        "fixed window" => fixed,
        "integral" => integral,
    ]
        @test all(abs.(fit_local(cost, x0) .- p_ref) .≤ p_err)
    end

    # Values of the other cost functions at autoScale's best fit, with no external reference;
    # they pin the current implementation against unintended changes.
    @test fixed(p_ref) ≈ 0.9232351196024403 rtol = 1e-10
    @test integral(p_ref) ≈ 0.8684558446640278 rtol = 1e-10
end
