# Benchmark of the error propagation: transform! and full cost evaluations.
#
# Run from the repository root with an environment that has the package, BenchmarkTools and
# Optim, e.g.
#
#     julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.add(["BenchmarkTools", "Optim"])'
#     julia --project=benchmark benchmark/error_propagation.jl
using Autoscalers, BenchmarkTools, Random, Optim, Printf
using Autoscalers: transform!

function synthetic(g; sizes, xs, σ, rng)
    L = Int[]; x = Float64[]; y = Float64[]
    for l in sizes, xi in xs
        push!(L, l); push!(x, xi); push!(y, g(l, xi) + σ * randn(rng))
    end
    return (; L, x, y, y_err=fill(σ, length(x)))
end
f(u) = 1 + tanh(u) + 0.3u
g(L, x) = L^(-0.25) * f(L * (x - 0.3))
std_data = synthetic(g; sizes=[8, 12, 16, 24, 32], xs=range(0.2, 0.4; length=161), σ=2e-3, rng=Xoshiro(42))
big_data = synthetic(g; sizes=[8, 12, 16, 24, 32, 48, 64, 96], xs=range(0.2, 0.4; length=1000), σ=2e-3, rng=Xoshiro(1))
bg = @ansatz (A, B, a, b, x_c) -> :L^b * ((:y ± :y_err) - A - B * (:x - x_c)) ~ :L^a * (:x - x_c)

cases = [
    ("805 pts, default", std_data, Autoscalers.DEFAULT_ANSATZ, [1.05, 0.22, 0.301]),
    ("805 pts, background", std_data, bg, [0.0, 0.0, 1.05, 0.22, 0.301]),
    ("8000 pts, default", big_data, Autoscalers.DEFAULT_ANSATZ, [1.05, 0.22, 0.301]),
]
costs = ["integral" => (;), "classic" => (; classic_cost_function=true),
         "fixed window" => (; eval_window=(0.28, 0.32))]

fmt(t) = t < 1e3 ? @sprintf("%.0f ns", t) : t < 1e6 ? @sprintf("%.1f μs", t / 1e3) : @sprintf("%.2f ms", t / 1e6)
println("| case | step | median time | allocs | memory |")
println("|---|---|---|---|---|")
for (name, data, ansatz, p) in cases
    s = Autoscaler(data, ansatz)
    b = @benchmark transform!($s, $p)
    println("| $name | `transform!` | ", fmt(median(b).time), " | ", b.allocs, " | ", round(b.memory / 1024; digits=1), " KiB |")
    for (cname, kw) in costs
        s = Autoscaler(data, ansatz; kw...)
        b = @benchmark $s($p)
        println("| $name | $cname cost | ", fmt(median(b).time), " | ", b.allocs, " | ", round(b.memory / 1024; digits=1), " KiB |")
    end
end
# a full fit: fixed number of Nelder-Mead iterations so both versions do the same work
s = Autoscaler(std_data; eval_window=(0.28, 0.32))
opts = Optim.Options(iterations=500, g_tol=0.0)
optimize(q -> s(q), [1.2, 0.1, 0.31], NelderMead(), opts)
b = @benchmark optimize(q -> $s(q), [1.2, 0.1, 0.31], NelderMead(), $opts) samples=20 evals=1
println("| 805 pts, default | fit, fixed window, 500 Nelder–Mead iterations | ", fmt(median(b).time), " | ", b.allocs, " | ", round(b.memory / 1024; digits=1), " KiB |")
