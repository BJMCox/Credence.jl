using Pkg
using TOML

# The prototype targets development APIs not yet present in General.
root = dirname(@__DIR__)
sources = TOML.parsefile(joinpath(root, "Project.toml"))["sources"]
packages = [
    Pkg.PackageSpec(; name, url=source["url"], rev=source["rev"]) for
    (name, source) in sources
]
Pkg.add(packages)
Pkg.instantiate()
