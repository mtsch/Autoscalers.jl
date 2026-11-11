using Documenter
using Autoscalers

DocMeta.setdocmeta!(Autoscalers, :DocTestSetup, :(using Autoscalers); recursive=true)

makedocs(;
    sitename="Autoscalers.jl",
    modules=[Autoscalers],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        edit_link="main",
    ),
    pages=["Home" => "index.md", "Internals" => "internals.md"],
    # Fails the build if an exported name has no docstring on a page.
    checkdocs=:exports,
)

# Pull requests from this repository are deployed to `previews/PR<number>`; the
# DocPreviewCleanup workflow removes them when the pull request is closed.
deploydocs(;
    repo="github.com/mtsch/Autoscalers.jl.git",
    devbranch="main",
    push_preview=true,
)
