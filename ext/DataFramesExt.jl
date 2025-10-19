module DataFramesExt

using Autoscalers
using DataFrames

DataFrames.DataFrame(a::Autoscaler, params...) = DataFrame(to_table(a, params...))

end
