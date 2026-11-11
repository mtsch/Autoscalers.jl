# # Usage guide
#
# This guide collapses the order parameter of two-dimensional site percolation,
# estimates the critical exponents with error bars, and tests one of them against its
# exact value. The data are the example of
# [autoScale](https://github.com/omelchert/autoScale): the relative size ``y`` of the
# largest cluster, with its error, at occupation probability ``p``, for system sizes
# ``L = 128, 256, 512``.

using Autoscalers
using CairoMakie
using DelimitedFiles
using Random

dir = joinpath(pkgdir(Autoscalers), "test", "data", "autoscale")
read_size(L) = readdlm(joinpath(dir, "orderParam_L$L.dat"); comments=true)
data = [
    (; L, p, y, y_err) for L in (128, 256, 512) for (p, y, y_err) in eachrow(read_size(L))
]
first(data, 3)

# Any [Tables.jl](https://github.com/JuliaData/Tables.jl) table works, e.g. a vector of
# named tuples as here, or a `DataFrame`.
#
# ## Fitting the collapse
#
# Near the percolation threshold ``p_c``, finite-size scaling predicts
# ``y = L^{-β/ν} f(L^{1/ν} (p - p_c))``. With ``a = 1/ν`` and ``b = β/ν``, the data
# collapse onto ``f`` when ``L^b y`` is plotted against ``L^a (p - p_c)``:

ansatz = @ansatz (a, b, p_c) -> :L^b * (:y ± :y_err) ~ :L^a * (:p - p_c)

# The scaling form holds close to ``p_c`` in the *unscaled* ``p``, so the points that
# are scored are chosen with `eval_window` in ``p``. It is narrow enough that every
# size covers it after scaling; otherwise [`fit_collapse`](@ref) would warn.

scaler = Autoscaler(data, ansatz; eval_window=(0.583, 0.602))
fit = fit_collapse(scaler, [0.7, 0.1, 0.59])
fit.params

# [`to_table`](@ref) gives the scaled data at the fitted parameters:

table = to_table(scaler, fit.params)

fig = Figure(; size=(800, 350))
unscaled = Axis(fig[1, 1]; xlabel=L"p", ylabel=L"y", title="unscaled")
scaled = Axis(fig[1, 2]; xlabel=L"L^a (p - p_c)", ylabel=L"L^b y", title="scaled")
vspan!(unscaled, 0.583, 0.602; color=(:gray, 0.2), label="eval_window")
for size in (128, 256, 512)
    rows = filter(r -> r.L == size, data)
    p, y, y_err = getindex.(rows, :p), getindex.(rows, :y), getindex.(rows, :y_err)
    errorbars!(unscaled, p, y, y_err)
    scatter!(unscaled, p, y; markersize=6, label="L = $size")
    on_size = table.L .== size
    errorbars!(scaled, table.x[on_size], table.y[on_size], table.y_err[on_size])
    scatter!(scaled, table.x[on_size], table.y[on_size]; markersize=6)
end
axislegend(unscaled; position=:lt)
fig

# ## Error bars
#
# [`bootstrap`](@ref) refits the collapse to synthetic data drawn from the fitted master
# curve with the measured errors. Here `n` is small to keep the documentation fast; use
# at least 1000 replicates in practice.

boot = bootstrap(scaler, fit.params; n=200, rng=Xoshiro(1))
confint(boot)

#-

stderror(boot)

# [`confint`](@ref) gives bias-corrected 95 % intervals, which can be asymmetric around
# the estimate.
#
# ## Testing an exponent
#
# For two-dimensional percolation ``ν = 4/3`` exactly, so ``a = 3/4``. The p-value of
# that hypothesis is

pvalue(boot, :a => 3 / 4)

# so the collapse is consistent with it. [`pvalue`](@ref) with `method=:null` refits
# under the hypothesis instead, which is slower but can test several parameters at once,
# e.g. `pvalue(boot, :a => 3 / 4, :b => 5 / 48; method=:null)`.
