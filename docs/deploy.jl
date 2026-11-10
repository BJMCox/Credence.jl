using Documenter

deploydocs(;
    root=@__DIR__,
    repo="github.com/JuliaBayes/Credence.jl.git",
    devbranch="main",
    push_preview=true,
    versions=["stable" => "v^", "v#.#.#", "dev" => "dev"],
)
