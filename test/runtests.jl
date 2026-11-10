using Autoscalers
using Autoscalers: lerp, interpolate, extrapolate, master_curve, get_parameters,
    free_parameter_names, window, transform!, select_subset, standard_master_curve, raw_x
using Optim
using Random
using Test

###
### Helpers
###

"""
    synthetic(g; sizes, xs, σ, rng)

Synthetic data set: `y = g(L, x)` for each `L` in `sizes` and `x` in `xs`, with constant
error bars `σ`. If `rng` is given, Gaussian noise of width `σ` is added.
"""
function synthetic(g; sizes=[8, 12, 16, 24, 32], xs, σ=1e-3, rng=nothing)
    L = Int[]
    x = Float64[]
    y = Float64[]
    for l in sizes, xi in xs
        push!(L, l)
        push!(x, xi)
        push!(y, g(l, xi) + (isnothing(rng) ? 0.0 : σ * randn(rng)))
    end
    return (; L, x, y, y_err=fill(σ, length(x)))
end

"""
    fit(scaler, x0)

Minimise the cost with Nelder–Mead, restarting once from the result.
"""
function fit(scaler, x0)
    opts = Optim.Options(iterations=20_000, g_tol=1e-14)
    res = optimize(p -> scaler(p), float.(x0), NelderMead(), opts)
    res = optimize(p -> scaler(p), Optim.minimizer(res), NelderMead(), opts)
    return Optim.minimizer(res), Optim.minimum(res)
end

std_(v) = sqrt(sum(abs2, v .- sum(v) / length(v)) / (length(v) - 1))

# names the `@ansatz` tests resolve in the calling module
ansatz_test_scale(L) = 2L
const ANSATZ_TEST_SHIFT = 0.5

const COSTS = [
    "integral" => (;),
    "classic" => (; classic_cost_function=true),
]

@testset "Autoscalers" begin

###
### Unit tests
###

@testset "interpolation" begin
    @test lerp((0.0, 0.0), (1.0, 2.0), 0.25) ≈ 0.5
    @test lerp((0.0, 0.0), (1.0, 2.0), 2.0) ≈ 4.0      # extrapolates
    @test lerp((1.0, 3.0), (1.0, 5.0), 1.0) == 3.0     # degenerate segment

    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 4.0]
    es = [0.1, 0.2, 0.3]
    @test interpolate(xs, ys, es, 1.0) == (1.0, 0.2)                # knot
    @test interpolate(xs, ys, es, 1.0; sq_errs=true)[2] ≈ 0.04
    y, e = interpolate(xs, ys, es, 1.5)
    @test y ≈ 2.5
    @test e ≈ 0.25
    @test interpolate(xs, ys, es, 1.5; sq_errs=true)[2] ≈ (0.04 + 0.09) / 2
    @test ismissing(interpolate(xs, ys, es, -0.1))
    @test ismissing(interpolate(xs, ys, es, 2.1))

    # extrapolate agrees with interpolate inside and is continuous at the edges
    @test extrapolate(xs, ys, es, 1.5)[1:2] == interpolate(xs, ys, es, 1.5; sq_errs=true)
    @test extrapolate(xs, ys, es, 1.5)[3] == 0
    for edge in (0.0, 2.0), ε in (1e-9, -1e-9)
        y0, dy0, _ = extrapolate(xs, ys, es, edge)
        y1, dy1, d = extrapolate(xs, ys, es, edge + ε)
        @test y1 ≈ y0 atol = 1e-8
        @test dy1 ≈ dy0 atol = 1e-8
        @test d ≤ 1e-9
    end
    @test all(extrapolate(xs, ys, es, 3.0) .≈ (7.0, 0.09, 0.5))   # slope 3, d = 1/2
    @test all(extrapolate(xs, ys, es, -1.0) .≈ (-1.0, 0.01, 0.5))
    @test_throws ArgumentError extrapolate([0.0], [0.0], [1.0], 0.0)

    # master curve: inverse-variance weighting, optional skip
    pts = [(1.0, 1.0), (3.0, 1.0), (10.0, 0.25)]
    Y, dY2 = master_curve(pts, 0)
    @test Y ≈ (1 + 3 + 40) / 6
    @test dY2 ≈ 1 / 6
    @test master_curve(pts, 3) == (2.0, 0.5)
end

@testset "@ansatz" begin
    a = @ansatz (a, b, x_c) -> :L^b * (:y ± :y_err) ~ :L^a * (:x - x_c)
    @test a.param_names == (:a, :b, :x_c)
    @test (a.x, a.size, a.errors, a.columns) == (:x, :L, (; y=:y_err), (:L, :x, :y))
    @test a.f((; L=4, x=2.0, y=3.0), (; a=0.5, b=1.0, x_c=1.0)) == (2.0, 12.0)
    @test sprint(show, a) ==
          "@ansatz((a, b, x_c)->:L ^ b * (:y ± :y_err) ~ :L ^ a * (:x - x_c))"

    # parameters keep the declared order; a single one needs no parentheses
    @test (@ansatz (x_c, b, a) -> :L^b * (:y ± :y_err) ~ :L^a * (:x - x_c)).param_names ==
          (:x_c, :b, :a)
    one = @ansatz x_c -> (:y ± :y_err) ~ :L * (:x - x_c)
    @test one.param_names == (:x_c,)
    @test one.f((; L=2, x=1.0, y=0.0), (; x_c=0.25)) == (1.5, 0.0)

    # x on the left-hand side (analytic background)
    bg = @ansatz (A, B, a, b, x_c) -> :L^b * ((:y ± :y_err) - A - B * (:x - x_c)) ~
        :L^a * (:x - x_c)
    p = (; A=1.0, B=2.0, a=0.0, b=0.0, x_c=1.0)
    @test bg.f((; L=1, x=2.0, y=10.0), p)[2] ≈ 10 - 1 - 2

    # other column names, the size column, and names that are not identifiers
    r = @ansatz (a, t_c) -> (:E ± :dE) ~ :N^a * (:t - t_c) size=:N
    @test (r.x, r.size, r.errors, r.columns) == (:t, :N, (; E=:dE), (:N, :t, :E))
    sp = @ansatz t_c -> (:var"the y" ± :var"y err") ~ :var"sys size" * (:t - t_c) size =
        :var"sys size"
    @test sp.size == Symbol("sys size")
    @test sp.errors == NamedTuple{(Symbol("the y"),)}((Symbol("y err"),))

    # x must be given when the right-hand side has several columns
    h = @ansatz a -> (:y ± :y_err) ~ :L^a * :x + :h x=:x
    @test (h.x, h.columns) == (:x, (:L, :x, :y, :h))
    @test_throws ArgumentError @macroexpand @ansatz a -> (:y ± :y_err) ~ :L^a * :x + :h

    # functions, constants and variables come from the calling scope; `$` copies a value
    # once
    shift = 1.0
    g = @ansatz a -> (:y ± :y_err) ~
        ansatz_test_scale(:L) * (:x - ANSATZ_TEST_SHIFT) + shift
    s = @ansatz a -> (:y ± :y_err) ~ :x - $shift + π * 0
    @test g.f((; L=2, x=1.0, y=0.0), (; a=0.0))[1] == 4 * 0.5 + 1.0
    @test s.f((; L=2, x=1.0, y=0.0), (; a=0.0))[1] == 0.0
    shift = 2.0
    @test g.f((; L=2, x=1.0, y=0.0), (; a=0.0))[1] == 4 * 0.5 + 2.0
    @test s.f((; L=2, x=1.0, y=0.0), (; a=0.0))[1] == 0.0

    # qualified names stay names, dotted calls still see columns
    q = @ansatz a -> (:y ± :y_err) ~ Base.sqrt(:L) * sqrt.(:x)
    @test q.f((; L=4, x=9.0, y=0.0), (; a=0.0))[1] == 6.0

    # `$` values and parameters also work as callees and as receivers of field access
    scale = x -> 2x
    config = (; shift=0.5)
    c = @ansatz a -> (:y ± :y_err) ~ ($scale)(:x) + ($scale).(:x) - ($config).shift
    @test c.f((; L=1, x=3.0, y=0.0), (; a=0.0))[1] == 6 + 6 - 0.5
    pc = @ansatz a -> (:y ± :y_err) ~ Base.Fix2(^, a)(:x)
    @test pc.f((; L=1, x=3.0, y=0.0), (; a=2.0))[1] == 9.0
    @test_throws ArgumentError @macroexpand @ansatz a -> (:y ± :e) ~ a.(:x)

    # several columns with errors
    two = @ansatz a -> (:y ± :y_err) - (:z ± :z_err) ~ :L^a * :x
    @test two.errors == (; y=:y_err, z=:z_err)

    # malformed input
    @test_throws ArgumentError @macroexpand @ansatz :L^b * :y ~ :L^a * (:x - x_c)
    @test_throws ArgumentError @macroexpand @ansatz L^b * y ~ L^a * (x - x_c)
    @test_throws ArgumentError @macroexpand @ansatz (a, a) -> (:y ± :e) ~ :x
    @test_throws ArgumentError @macroexpand @ansatz (a, 1) -> (:y ± :e) ~ :x
    @test_throws ArgumentError @macroexpand @ansatz a -> (:y ± :e) + :x
    @test_throws ArgumentError @macroexpand @ansatz a -> :y ~ :x
    @test_throws ArgumentError @macroexpand @ansatz a -> (:y ± :e) ~ (:x ± :xe)
    @test_throws ArgumentError @macroexpand @ansatz a -> (:y ± :e) ~ :x + :y
    @test_throws ArgumentError @macroexpand @ansatz a -> (:y ± :e) + (:y ± :e) ~ :x
    @test_throws ArgumentError @macroexpand @ansatz a -> (:y ± e) ~ :x
    @test_throws ArgumentError @macroexpand @ansatz a -> (:y ± :e) ~ a(:x)
    @test_throws ArgumentError @macroexpand @ansatz a -> (:y ± :e) ~ :x foo=:bar
    @test_throws ArgumentError @macroexpand @ansatz a -> (:y ± :e) ~ :x size=L
end

@testset "Ansatz (function form)" begin
    a = Ansatz((:a, :b, :t_c); x=:t, errors=(; E=:dE), columns=(:h,), size=:N) do row, p
        return row.N^p.a * (row.t - p.t_c), row.N^p.b * row.E + row.h
    end
    @test a.param_names == (:a, :b, :t_c)
    @test (a.x, a.size, a.errors) == (:t, :N, (; E=:dE))
    @test a.columns == (:N, :t, :h, :E)
    @test startswith(sprint(show, a), "Ansatz(")

    # the same scaling as the default ansatz gives the same cost
    g(L, x) = L^-0.25 * (1 + tanh(L * (x - 0.3)))
    data = synthetic(g; sizes=[8, 16, 32], xs=range(0.2, 0.4; length=41), σ=1e-3)
    renamed = (; N=data.L, t=data.x, E=data.y, dE=data.y_err, h=zero(data.x))
    p = [1.1, 0.2, 0.31]
    for kw in ((;), (; classic_cost_function=true), (; eval_window=(0.28, 0.32)))
        @test Autoscaler(renamed, a; kw...)(p) == Autoscaler(data; kw...)(p)
    end

    # only the columns used by the ansatz are read: a missing value elsewhere is ignored
    extra = (; renamed..., unused=fill(missing, length(data.x)))
    @test @test_logs(Autoscaler(extra, a)(p)) == Autoscaler(renamed, a)(p)

    # several columns with errors: the uncertainties combine
    two = Ansatz((:a,); x=:x, errors=(; y=:y_err, z=:z_err)) do row, p
        return row.L^p.a * row.x, row.y - row.z
    end
    s = Autoscaler((; data..., z=zero(data.y), z_err=data.y_err), two)
    transform!(s, [1.0])
    @test s.y_err_scaled ≈ sqrt(2) .* data.y_err

    # nonlinear propagation: σ(y²) = 2|y|σ(y), σ(y z) = sqrt((zσ_y)² + (yσ_z)²)
    sq = Ansatz((:a,); x=:x, errors=(; y=:y_err, z=:z_err)) do row, p
        return row.x, row.y^2 + row.y * row.z
    end
    zs = (; data..., z=fill(2.0, length(data.y)), z_err=fill(0.5, length(data.y)))
    s = Autoscaler(zs, sq)
    transform!(s, [0.0])
    y, σy, z, σz = s.data.y, last(s.errors.y), s.data.z, last(s.errors.z)
    @test s.y_err_scaled ≈ sqrt.(((2 .* y .+ z) .* σy) .^ 2 .+ (y .* σz) .^ 2)

    # the size column is passed as an integer
    int_size = Ansatz((); x=:x, errors=(; y=:y_err)) do row, p
        row.L isa Int || error("expected an integer size")
        return row.x, row.y
    end
    @test Autoscaler(data, int_size)() isa Float64

    # invalid definitions
    f(row, p) = (row.x, row.y)
    @test_throws ArgumentError Ansatz(f, (:a, :a); x=:x, errors=(; y=:y_err))
    @test_throws ArgumentError Ansatz(f, (:a,); x=:x, errors=(;))
    @test_throws ArgumentError Ansatz(f, (:a,); x=:x, errors=(:y, :y_err))
    @test_throws ArgumentError Ansatz(f, (:a,); x=:L, errors=(; y=:y_err))
    @test_throws ArgumentError Ansatz(f, (:a,); x=:x, errors=(; x=:x_err))
    @test_throws ArgumentError Ansatz(f, (:a,); x=:x, errors=(; y=:x))

    # the scaled x must not carry an error, the scaled y must
    s = Autoscaler(data, Ansatz((row, p) -> (row.y, row.y), (); x=:x, errors=(; y=:y_err)))
    @test_throws ArgumentError s()
    s = Autoscaler(data, Ansatz((row, p) -> (row.x, row.x), (); x=:x, errors=(; y=:y_err)))
    @test_throws ArgumentError s()
end

@testset "Autoscaler construction" begin
    data = synthetic((L, x) -> x; sizes=[4, 6], xs=0:0.1:1)

    s = Autoscaler(data)
    @test s.avail_sizes == [4, 6]
    @test s.size_map == Dict(4 => 1:11, 6 => 12:22)
    @test free_parameter_names(s) == (:a, :b, :x_c)
    @test startswith(sprint(show, s), "Autoscaler(@ansatz((a, b, x_c)->")

    # rows are sorted by (L, x) regardless of input order
    perm = randperm(Xoshiro(1), length(data.x))
    shuffled = map(v -> v[perm], data)
    s2 = Autoscaler(shuffled)
    @test raw_x(s2) == raw_x(s)
    @test s2.sizes == s.sizes

    # zero-error rows are dropped with a warning, and the eval mask stays aligned
    bad = (; L=data.L, x=data.x, y=data.y, y_err=copy(data.y_err))
    bad.y_err[[1, 2, 15]] .= 0
    s3 = @test_logs (:warn, r"3 rows with non-positive") match_mode = :any Autoscaler(
        bad; eval_window=(0.25, 0.75)
    )
    @test length(raw_x(s3)) == 19
    @test length(s3.eval_mask) == 19
    @test all(0.25 .≤ raw_x(s3)[s3.eval_mask] .≤ 0.75)
    @test count(s3.eval_mask) == 9

    # hardcoded parameters
    s4 = Autoscaler(data; fix=(; a=1.5))
    @test free_parameter_names(s4) == (:b, :x_c)
    @test get_parameters(s4, [0.1, 0.2]) == [1.5, 0.1, 0.2]
    @test_throws ArgumentError get_parameters(s4, [0.1])
    @test_throws ArgumentError s4(0.1)
    p1 = get_parameters(s4, [0.1, 0.2])
    get_parameters(s4, [0.3, 0.4])
    @test p1 == [1.5, 0.1, 0.2]

    # remapped column names come from the ansatz
    renamed = (; size=data.L, t=data.x, val=data.y, val_err=data.y_err)
    ans = @ansatz (a, b, x_c) -> :size^b * (:val ± :val_err) ~ :size^a * (:t - x_c) size =
        :size
    s5 = Autoscaler(renamed, ans)
    @test raw_x(s5) == raw_x(s)
    @test s5.data.val == s.data.y

    # invalid input
    @test_throws ArgumentError Autoscaler(data; fix=(; c=1.0))
    @test_throws ArgumentError Autoscaler(data; fix=Dict(:a => 1.0))
    @test_throws ArgumentError Autoscaler(synthetic((L, x) -> x; sizes=[4], xs=0:0.1:1))
    @test_throws ArgumentError Autoscaler(data; eval_window=(2.0, 3.0))
    @test_throws ArgumentError Autoscaler(data; eval_window=(0.5, 0.4))
    @test_throws ArgumentError Autoscaler(
        data; eval_window=(0.2, 0.4), classic_cost_function=true
    )
    @test_throws ArgumentError Autoscaler(data; window_width=0)
    @test_logs (:warn, r"L=6 has no data") Autoscaler(
        synthetic((L, x) -> x; sizes=[4, 6], xs=0:0.1:1) |>
        d -> (; L=d.L, x=d.x .+ (d.L .== 6) .* 10, y=d.y, y_err=d.y_err);
        eval_window=(0.0, 1.0),
    )
end

@testset "Autoscaler input validation" begin
    data = synthetic((L, x) -> x; sizes=[4, 6], xs=0:0.1:1)
    allowmissing(v) = Vector{Union{Missing,eltype(v)}}(v)

    m = map(allowmissing, data)
    m.y[3] = missing
    m.x[14] = missing
    s = @test_logs (:warn, r"2 rows with missing") Autoscaler(m)
    @test length(raw_x(s)) == 20

    nf = map(copy, data)
    nf.y[1] = NaN
    nf.y_err[2] = Inf
    nf.x[20] = -Inf
    s = @test_logs (:warn, r"3 rows with non-finite") Autoscaler(nf)
    @test all(isfinite, s.data.y)
    @test all(isfinite, last(s.errors.y))
    @test length(raw_x(s)) == 19

    neg = map(copy, data)
    neg.y_err[5] = -0.1
    s = @test_logs (:warn, r"1 rows with non-positive") Autoscaler(neg)
    @test all(>(0), last(s.errors.y))

    # each kind is reported on every construction, not just the first one
    @test_logs (:warn, r"1 rows with non-positive") Autoscaler(neg)

    # system sizes
    @test Autoscaler((; data..., L=float.(data.L))).avail_sizes == [4, 6]
    @test_throws ArgumentError Autoscaler((; data..., L=data.L .+ 0.5))
    @test_throws ArgumentError Autoscaler((; data..., L=data.L .- 4))

    # every size needs two usable points, also after filtering
    one = (; L=[data.L; 8], x=[data.x; 0.5], y=[data.y; 0.5], y_err=[data.y_err; 1e-3])
    @test_throws ArgumentError Autoscaler(one)
    thin = map(copy, data)
    thin.y_err[13:22] .= 0
    @test_throws ArgumentError Autoscaler(thin)

    # duplicate (L, x) rows, also when they are not adjacent in the input
    dup = (; L=[data.L; 6], x=[data.x; 0.3], y=[data.y; 0.3], y_err=[data.y_err; 1e-3])
    @test_throws ArgumentError Autoscaler(dup)

    # scaled x must be ascending within each size
    rev = @ansatz (a, b, x_c) -> :L^b * (:y ± :y_err) ~ :L^a * (x_c - :x)
    s = Autoscaler(data, rev)
    @test_throws ArgumentError s(1.0, 0.0, 0.5)
    @test_throws ArgumentError to_table(s, 1.0, 0.0, 0.5)
end

@testset "classic cost: Houdayer & Hartmann reference" begin
    # Reference values from an independent matrix least-squares fit (numpy) of the points
    # selected according to Appendix A of PRB 70, 014418. With a = b = x_c = 0 the
    # scaling is the identity.
    data = (;
        L=[4, 4, 4, 6, 6, 8, 8],
        x=[0.0, 1.0, 2.0, 0.5, 2.0, 1.0, 3.0],
        y=[0.0, 1.0, 2.0, 0.4, 2.2, 1.1, 2.9],
        y_err=[0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.3],
    )
    s = Autoscaler(data; classic_cost_function=true)
    transform!(s, [0.0, 0.0, 0.0])

    # (L, x) => (number of selected points, Y, dY²); nothing brackets L=4, x=0 or L=8, x=3
    expected = [
        (4, 1.0) => (4, 1.065718799368088, 0.006887835703001581),
        (4, 2.0) => (4, 2.088783570300158, 0.015229067930489736),
        (6, 0.5) => (2, 0.5, 0.005),
        (6, 2.0) => (4, 1.992592592592594, 0.007037037037037043),
        (8, 1.0) => (4, 0.98139534883721, 0.006511627906976748),
    ]
    for ((L, x), (n, Y, dY2)) in expected
        @test length(select_subset(s, x, L)[1]) == n
        m = standard_master_curve(s, x, L)
        @test m[1] ≈ Y
        @test m[2] ≈ dY2
    end
    @test ismissing(standard_master_curve(s, 0.0, 4))
    @test ismissing(standard_master_curve(s, 3.0, 8))
    @test s(0.0, 0.0, 0.0) ≈ 0.5113810925012936

    # x = 2 coincides with the last point of L=6: the closed bracket x[j] ≤ x ≤ x[j + 1]
    # selects (0.5, 2.0) from L=6, while L=8 brackets it with (1.0, 3.0).
    @test select_subset(s, 2.0, 4)[1] == [0.5, 2.0, 1.0, 3.0]
    # on an interior knot, the pair starting at the knot is used
    @test select_subset(s, 1.0, 6)[1] == [1.0, 2.0, 1.0, 3.0]
end

@testset "window" begin
    data = synthetic((L, x) -> x; sizes=[4, 6], xs=-1:0.1:3)
    s = Autoscaler(data; window_width=0.5)
    transform!(s, [0.0, 0.0, 0.0])
    lo, hi = window(s)
    # width 4 shrunk to 2; the side further from zero (hi) is shrunk first
    @test hi - lo ≈ 2
    @test (lo, hi) == (-1.0, 1.0)
    s = Autoscaler(data)
    transform!(s, [0.0, 0.0, 0.0])
    @test window(s) == (-1.0, 3.0)
end

@testset "to_table and cost_breakdown" begin
    g(L, x) = tanh(L * (x - 0.5))
    data = synthetic(g; sizes=[4, 6, 8], xs=0:0.02:1)
    for kw in ((;), (; classic_cost_function=true), (; eval_window=(0.4, 0.6)))
        s = Autoscaler(data; kw...)
        t = to_table(s, 1.0, 0.0, 0.5)
        @test keys(t) == (:L, :x, :y, :y_err, :master_curve, :in_window)
        @test length(t.x) == length(data.x)
        @test t.x ≈ s.sizes .* (raw_x(s) .- 0.5)
        @test to_table(s, [1.0, 0.0, 0.5]).y == t.y
        inner = abs.(t.x) .< 1
        @test all(isfinite, t.master_curve[inner])
        @test maximum(abs, (t.y.-t.master_curve)[inner]) < 0.05
    end
    # the master curve a point is compared to must leave out its own size
    two = (; L=[4, 4, 4, 6, 6, 6], x=[0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
             y=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], y_err=fill(0.1, 6))
    for kw in ((;), (; classic_cost_function=true), (; eval_window=(0.0, 1.0)))
        t = to_table(Autoscaler(two; kw...), 0.0, 0.0, 0.0)
        m4 = filter(isfinite, t.master_curve[t.L.==4])
        m6 = filter(isfinite, t.master_curve[t.L.==6])
        @test length(m4) ≥ 2 && all(≈(1.0), m4)
        @test length(m6) ≥ 2 && all(≈(0.0), m6)
    end
    t = to_table(Autoscaler(two; use_all_sizes=true), 0.0, 0.0, 0.0)
    @test t.master_curve ≈ fill(0.5, 6)

    s = Autoscaler(data)
    @test_throws ArgumentError cost_breakdown(s, 1.0, 0.0, 0.5)
    s = Autoscaler(data; eval_window=(0.4, 0.6))
    b = cost_breakdown(s, 1.0, 0.0, 0.5)
    @test b.cost ≈ b.chi2 + b.penalty
    @test b.penalty == 0
    @test b.cost == s(1.0, 0.0, 0.5)
end

###
### Synthetic scaling forms: the optimiser must recover the true parameters.
###

@testset "recovery: standard power-law collapse" begin
    a0, b0, xc0 = 1.0, 0.25, 0.3
    f(u) = 1 + tanh(u) + 0.3u
    g(L, x) = L^(-b0) * f(L^a0 * (x - xc0))
    xs = range(xc0 - 0.1, xc0 + 0.1; length=161)
    x0 = [1.2, 0.1, 0.31]
    # coverage bound: w ≤ 0.1 / (32/8)^a0 = 0.025
    for (name, kw) in [COSTS; "fixed window" => (; eval_window=(xc0 - 0.025, xc0 + 0.025))]
        @testset "$name, exact data" begin
            s = Autoscaler(synthetic(g; xs); kw...)
            @test s(a0, b0, xc0) < 0.05
            @test s(a0, b0, xc0) < s(a0 + 0.1, b0, xc0)
            @test s(a0, b0, xc0) < s(a0, b0 + 0.1, xc0)
            @test s(a0, b0, xc0) < s(a0, b0, xc0 + 0.005)
            p, c = fit(s, x0)
            @test p[1] ≈ a0 atol = 0.02
            @test p[2] ≈ b0 atol = 0.02
            @test p[3] ≈ xc0 atol = 1e-3
        end
        @testset "$name, noisy data" begin
            s = Autoscaler(synthetic(g; xs, σ=2e-3, rng=Xoshiro(42)); kw...)
            p, c = fit(s, x0)
            @test p[1] ≈ a0 atol = 0.1
            @test p[2] ≈ b0 atol = 0.05
            @test p[3] ≈ xc0 atol = 3e-3
            @test 0.3 < c < 2       # χ²-like: ≈ 1 when errors are correct
        end
    end

    @testset "f(0) = 0: a narrow window only identifies a - b" begin
        # Near x_c, y ≈ f'(0) L^(a-b) (x - x_c), so a and b are degenerate without
        # curvature in the window. The combination a - b must still come out right.
        h(L, x) = L^(-b0) * tanh(L^a0 * (x - xc0))
        s = Autoscaler(synthetic(h; xs, σ=2e-3, rng=Xoshiro(42));
                       eval_window=(xc0 - 0.02, xc0 + 0.02))
        p, _ = fit(s, x0)
        @test p[1] - p[2] ≈ a0 - b0 atol = 0.05
        @test p[3] ≈ xc0 atol = 3e-3
    end
end

@testset "recovery: scale-invariant crossing (b = 0)" begin
    a0, xc0 = 0.75, 0.06
    # paper-like: weak L-dependence, S-shaped master curve, sizes 5..10
    g(L, x) = -0.02 - 0.01 * atan(40 * L^a0 * (x - xc0))
    xs = range(0.05, 0.07; length=81)
    data = synthetic(g; sizes=5:10, xs, σ=2e-4, rng=Xoshiro(7))
    # coverage bound: w ≤ 0.01 / 2^a0 ≈ 0.006
    for (name, kw) in [COSTS; "fixed window" => (; eval_window=(0.0545, 0.0655))]
        @testset "$name" begin
            s = Autoscaler(data; kw...)
            p, c = fit(s, [1.0, 0.1, 0.061])
            @test p[1] ≈ a0 atol = 0.1
            @test p[2] ≈ 0 atol = 0.05
            @test p[3] ≈ xc0 atol = 1e-3
            @test c < 2
        end
    end

    @testset "fixed window: a too narrow window loses precision" begin
        # Not a bug, but a property worth knowing: with w well below the coverage bound,
        # b and x_c become correlated and scatter much more between noise realisations.
        spread(w) = std_(map(7:12) do seed
            d = synthetic(g; sizes=5:10, xs, σ=2e-4, rng=Xoshiro(seed))
            fit(Autoscaler(d; eval_window=(xc0 - w, xc0 + w)), [1.0, 0.1, 0.061])[1][2]
        end)
        @test spread(0.002) > spread(0.0055)
    end
end

@testset "recovery: hardcoded exponent" begin
    a0, b0, xc0 = 1.5, -0.5, 0.0
    g(L, x) = L^(-b0) * exp(-(L^a0 * x - 1)^2)
    xs = range(-0.1, 0.3; length=201)
    data = synthetic(g; sizes=[4, 6, 8, 10], xs, σ=1e-3)
    s = Autoscaler(data; fix=(; a=a0), eval_window=(0.0, 0.1))
    @test free_parameter_names(s) == (:b, :x_c)
    p, _ = fit(s, [-0.3, 0.01])
    @test p[1] ≈ b0 atol = 0.02
    @test p[2] ≈ xc0 atol = 2e-3
end

@testset "recovery: analytic background + critical part" begin
    inv_ν, y0 = 1.48876, 3 - 2 * 1.48876
    A0, B0, xc0 = 0.29, -0.75, 0.0597
    f(u) = 0.01 * tanh(5u)
    g(L, x) = A0 + B0 * (x - xc0) + L^(-y0) * f(L^inv_ν * (x - xc0))
    xs = range(0.045, 0.075; length=121)
    # coverage bound: w ≤ 0.015 / 2^1.49 ≈ 0.0054
    data = synthetic(g; sizes=5:10, xs, σ=5e-5, rng=Xoshiro(3))
    win = (xc0 - 0.004, xc0 + 0.004)

    @testset "fixed exponents (literals in the ansatz)" begin
        ans = @ansatz (A, B, x_c) -> :L^0.0225 * ((:y ± :y_err) - A - B * (:x - x_c)) ~
            :L^1.48876 * (:x - x_c)
        @test ans.param_names == (:A, :B, :x_c)
        s = Autoscaler(data, ans; eval_window=win)
        p, c = fit(s, [0.28, -0.6, 0.0605])
        @test p[2] ≈ B0 atol = 0.05
        @test p[3] ≈ xc0 atol = 5e-4
        @test c < 2
        # A is only weakly constrained: L^0.0225 is nearly constant, so a shift of A is
        # almost indistinguishable from a constant in the master curve.
        @test p[1] ≈ A0 atol = 0.02
        @test cost_breakdown(s, p).penalty < 0.1
    end

    @testset "free crossover exponent" begin
        ans = @ansatz (A, B, a, b, x_c) -> :L^b * ((:y ± :y_err) - A - B * (:x - x_c)) ~
            :L^a * (:x - x_c)
        s = Autoscaler(data, ans; fix=(; b=y0), eval_window=win)
        @test free_parameter_names(s) == (:A, :B, :a, :x_c)
        p, _ = fit(s, [0.29, -0.7, 1.3, 0.0600])
        @test p[3] ≈ inv_ν atol = 0.1
        @test p[4] ≈ xc0 atol = 5e-4
    end

    @testset "simple ansatz on background data gives an effective exponent" begin
        # This is the situation discussed for the paper: fitting L^b y ~ L^a (x - x_c)
        # to data with a sizeable analytic background yields a < 1/ν and a worse cost.
        bgdata = (; data.L, data.x, y=data.y .- A0, data.y_err)
        simple = Autoscaler(bgdata; eval_window=win)
        p_s, c_s = fit(simple, [1.0, 0.0, 0.0600])
        ans = @ansatz (A, B, x_c) -> :L^0.0225 * ((:y ± :y_err) - A - B * (:x - x_c)) ~
            :L^1.48876 * (:x - x_c)
        full = Autoscaler(bgdata, ans; eval_window=win)
        _, c_f = fit(full, [0.0, -0.6, 0.0605])
        @test p_s[1] < inv_ν - 0.2
        @test c_s > c_f
    end
end

###
### Behaviour of the fixed-window cost
###

@testset "fixed-window cost is continuous and cannot be gamed" begin
    a0, b0, xc0 = 1.0, 0.0, 0.3
    g(L, x) = L^(-b0) * tanh(L^a0 * (x - xc0))
    xs = range(xc0 - 0.1, xc0 + 0.1; length=81)
    data = synthetic(g; xs, σ=1e-3, rng=Xoshiro(11))
    s = Autoscaler(data; eval_window=(xc0 - 0.03, xc0 + 0.03))

    # scan a through the point where the largest size stops covering the window
    as = range(0.5, 2.5; length=2001)
    cs = [s(a, b0, xc0) for a in as]
    @test all(isfinite, cs)
    jumps = abs.(diff(cs))
    @test maximum(jumps) < 0.05 * maximum(cs)

    # large exponents push the data outside each other's range: penalised, not rewarded
    @test cost_breakdown(s, 2.5, b0, xc0).penalty > 0
    @test s(2.5, b0, xc0) > s(a0, b0, xc0)
    # absurd shifts of x_c are penalised as well
    @test s(a0, b0, xc0 + 1.0) > 100 * s(a0, b0, xc0)

    # the scored set of points never changes with the parameters
    n_before = count(s.eval_mask)
    s(2.0, 0.5, 0.2)
    @test count(s.eval_mask) == n_before
end

include("autoscale_reference.jl")

end # @testset "Autoscalers"
