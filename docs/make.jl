using Documenter
using Autoscalers

makedocs(;
    sitename="Autoscalers.jl",
    modules=[Autoscalers],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        edit_link="main",
    ),
    pages=["Home" => "index.md"],
    # Fails the build if an exported name has no docstring on a page.
    checkdocs=:exports,
)

deploydocs(; repo="github.com/mtsch/Autoscalers.jl.git", devbranch="main")
