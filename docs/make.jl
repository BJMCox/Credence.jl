using Credence
using Documenter
using DocumenterCodeBlocks
using Literate

root = dirname(@__DIR__)
for name in ("signal_background", "scalar", "wren")
    options = if name == "wren"
        (; codefence="```julia" => "```")
    else
        (;)
    end
    Literate.markdown(
        joinpath(root, "examples", "$name.jl"),
        joinpath(@__DIR__, "src", "tutorials");
        credit=false,
        options...,
    )
end
cp(
    joinpath(root, "assets", "logo.svg"),
    joinpath(@__DIR__, "src", "assets", "logo.svg");
    force=true,
)

makedocs(;
    root=@__DIR__,
    sitename="Credence.jl",
    repo=Documenter.Remotes.GitHub("BJMCox", "Credence.jl"),
    modules=[Credence],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bjmcox.github.io/Credence.jl/stable/",
        edit_link="main",
        inventory_version="",
        assets=["assets/logo.css", "assets/favicon.ico"],
    ),
    pages=[
        "Getting started" => "index.md",
        "Tutorials" => [
            "Scientific model" => "tutorials/signal_background.md",
            "Programmatic elicitation" => "tutorials/scalar.md",
            "Guided REPL session" => "tutorials/guided.md",
            "A joint prior in Wren" => "tutorials/wren.md",
        ],
        "Guide" => ["Beliefs and validation" => "validation.md"],
        "Reference" => "api.md",
    ],
    plugins=[CodeBlocks()],
    checkdocs=:exports,
    warnonly=false,
)
