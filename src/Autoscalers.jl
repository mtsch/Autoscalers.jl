module Autoscalers

using Tables
using ForwardDiff: ForwardDiff, Dual

export Ansatz, Autoscaler, @ansatz, to_table, cost_breakdown

include("ansatz.jl")
include("autoscaler.jl")
include("interpolation.jl")
include("cost-integral.jl")
include("cost-classic.jl")
include("cost-fixed-window.jl")

end
