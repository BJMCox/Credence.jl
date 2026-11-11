using Credence
using Test
using Random, MeasureBase, Distributions, PDMats

include("core.jl")
include("discrete.jl")
include("optimization.jl")
include("inputs.jl")
include("integration.jl")

include("flatppl.jl")
include("judgment_diagnostics.jl")
include("sampled_comparison.jl")
include("reporting.jl")
include("families.jl")
include("predictive.jl")
include("persistence.jl")
include("family_interop.jl")
include("compound/horseshoe.jl")
include("compound/r2d2.jl")
include("compound/dirichlet_laplace.jl")
include("compound/group_horseshoe.jl")
include("compound/functional_horseshoe.jl")
include("compound/multiplicative_gamma.jl")
include("compound/normal_gamma.jl")
include("compound/generalized_double_pareto.jl")
include("compound/integration.jl")
include("compound/flatppl.jl")
include("compound/simplex.jl")
