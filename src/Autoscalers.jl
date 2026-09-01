module Autoscalers

using Tables
using Measurements
export Autoscaler, to_table, @ansatz

include("ansatz.jl")
include("autoscaler.jl")
include("interpolated-curve.jl")

include("new-cost.jl")
include("old-cost.jl")

end
