using Credence
using Documenter
using DocumenterCodeBlocks
using Literate

root = dirname(@__DIR__)
for name in ("scalar", "wren")
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
        assets=["assets/favicon.ico"],
    ),
    pages=[
        "Getting started" => "index.md",
        "Tutorials" => [
            "Comparing prior families" => "tutorials/scalar.md",
            "A joint prior in Wren" => "tutorials/wren.md",
        ],
        "Guide" => ["Beliefs and validation" => "validation.md"],
        "Reference" => "api.md",
    ],
    plugins=[CodeBlocks()],
    checkdocs=:exports,
    warnonly=false,
)
