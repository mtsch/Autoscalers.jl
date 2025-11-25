module DataFramesExt

using Autoscalers
using DataFrames

DataFrames.DataFrame(a::Autoscaler, params::Union{Vector,Tuple}) = DataFrame(to_table(a, params))

end
