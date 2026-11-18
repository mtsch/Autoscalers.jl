module Autoscalers

using Tables
using ForwardDiff: ForwardDiff, Dual
using Optim: Optim, NelderMead, optimize
using ProgressMeter: Progress, finish!, next!
using QuadGK: alloc_segbuf, quadgk
using Random: Random, Xoshiro
using SpecialFunctions: erfc, erfcinv
using Statistics: quantile, std
using StatsAPI: StatsAPI, confint, pvalue, stderror

export Ansatz, Autoscaler, @ansatz, to_table
export fit_collapse, bootstrap, BootstrapResult, confint, stderror, pvalue, goodness_of_fit

include("ansatz.jl")
include("autoscaler.jl")
include("interpolation.jl")
include("cost-classic.jl")
include("cost-fixed-window.jl")
include("cost-integral-window.jl")
include("fit.jl")
include("bootstrap.jl")

end
