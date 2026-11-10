module Autoscalers

using Tables
using ForwardDiff: ForwardDiff, Dual
using Optim: Optim, NelderMead, optimize
using ProgressMeter: Progress, finish!, next!
using Random: Random, Xoshiro
using Statistics: quantile, std
using StatsAPI: StatsAPI, confint, pvalue, stderror

export Ansatz, Autoscaler, @ansatz, to_table, cost_breakdown
export fit_collapse, bootstrap, BootstrapResult, confint, stderror, pvalue

include("ansatz.jl")
include("autoscaler.jl")
include("interpolation.jl")
include("cost-integral.jl")
include("cost-classic.jl")
include("cost-fixed-window.jl")
include("fit.jl")
include("bootstrap.jl")

end
