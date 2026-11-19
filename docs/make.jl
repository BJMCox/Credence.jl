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

examples = [
    "Easy" =>
        ["Calibration tolerance" => "zero_offset", "Pump repair time" => "pump_repair"],
    "Intermediate" => [
        "Monitor calibration" => "monitor_calibration",
        "Water-source fractions" => "water_sources",
    ],
    "Advanced" => ["Signal and background" => "scalar", "Wren integration" => "wren"],
    "Very advanced" => [
        "Shrinkage for building load" => "shrinkage",
        "FlatPPL integration" => "flatppl",
    ],
]

for (_, group) in examples, (_, name) in group
    preprocess = identity
    if name == "scalar"
        preprocess = self_contained
    end
    Literate.markdown(
        joinpath(root, "examples", "$name.jl"),
        joinpath(@__DIR__, "src", "tutorials");
        credit=false,
        preprocess=preprocess,
    )
end

makedocs(;
    root=@__DIR__,
    sitename="Credence.jl",
    repo=Documenter.Remotes.GitHub("BJMCox", "Credence.jl"),
    modules=[Credence],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bjmcox.github.io/Credence.jl/dev/",
        edit_link="main",
        inventory_version="",
        assets=["assets/favicon.ico"],
    ),
    pages=[
        "Getting started" => "index.md",
        "Examples" => [
            level => [title => "tutorials/$name.md" for (title, name) in group] for
            (level, group) in examples
        ],
        "Guide" => [
            "Named inputs" => "inputs.md",
            "Candidate families" => "families.md",
            "Beliefs and validation" => "validation.md",
        ],
        "Tilting an existing prior" => "tilting.md",
        "Reference" => "api.md",
    ],
    plugins=[CodeBlocks()],
    checkdocs=:exports,
    warnonly=false,
)
