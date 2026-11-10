using Credence
using Documenter

makedocs(;
    root=@__DIR__,
    sitename="Credence.jl",
    repo=Documenter.Remotes.GitHub("JuliaBayes", "Credence.jl"),
    modules=[Credence],
    format=Documenter.HTML(; edit_link="main"),
    pages=["Home" => "index.md"],
    checkdocs=:exports,
    warnonly=false,
)

deploydocs(;
    repo="github.com/JuliaBayes/Credence.jl.git",
    devbranch="main",
    push_preview=true,
)
