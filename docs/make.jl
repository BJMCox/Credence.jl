using Credence
using Documenter
using DocumenterCodeBlocks
using Literate

root = dirname(@__DIR__)

function self_contained(source)
    for name in ("signal_background", "feedback")
        included = read(joinpath(root, "examples", "$name.jl"), String)
        if name == "signal_background"
            included = replace(included, "# ## " => "# ### ")
            included =
                replace(included, r"^# # .*\n" => "# ## Model and candidate definitions\n")
        else
            included = "# ## Plotting functions\n#\n" * included
        end
        source = replace(
            source,
            "include(joinpath(pkgdir(Credence), \"examples\", \"$name.jl\"))" =>
                included,
        )
    end
    return source
end

for name in ("scalar", "wren", "flatppl")
    options = if name == "wren"
        (; codefence="```julia" => "```")
    else
        (;)
    end
    preprocess = identity
    if name == "scalar"
        preprocess = self_contained
    end
    Literate.markdown(
        joinpath(root, "examples", "$name.jl"),
        joinpath(@__DIR__, "src", "tutorials");
        credit=false,
        preprocess=preprocess,
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
        "Examples" => [
            "Base elicitation" => "tutorials/scalar.md",
            "Wren integration" => "tutorials/wren.md",
            "FlatPPL integration" => "tutorials/flatppl.md",
        ],
        "Guide" =>
            ["Named inputs" => "inputs.md", "Beliefs and validation" => "validation.md"],
        "Reference" => "api.md",
    ],
    plugins=[CodeBlocks()],
    checkdocs=:exports,
    warnonly=false,
)
