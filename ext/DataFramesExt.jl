module DataFramesExt

using Autoscalers
using DataFrames

function DataFrames.DataFrame(a::Autoscaler, params::Union{Vector,Tuple})
    return DataFrame(to_table(a, params))
end

end
