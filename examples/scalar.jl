# # [Comparing prior families](@id scalar-tutorial)
#
# Suppose an expert places the lower and upper quartiles at -0.5 and 1.5.
# Fit a Normal family and a Student-t family with five degrees of freedom.
# Both can match these quartiles, while assigning different probabilities to extreme values.
#
# Run this file from the package directory with Julia 1.13 or later:
#
# ```sh
# julia --project=examples -e 'using Pkg; Pkg.instantiate()'
# julia --project=examples examples/scalar.jl
# ```
#
# ## Define the families
#
using Credence
using Distributions: LocationScale, Normal, TDist
using MeasureBase
using Optimization
using OptimizationOptimJL
using Random: Xoshiro

# Work in log scale so every proposed scale remains positive.
candidates = (
    Candidate(
        :normal,
        p -> AbstractMeasure(Normal(p[1], exp(p[2]))),
        ([0.0, 0.0], [1.0, -0.5]),
    ),
    Candidate(
        :student_t,
        p -> AbstractMeasure(LocationScale(p[1], exp(p[2]), TDist(5))),
        ([0.0, 0.0], [1.0, -0.5]),
    ),
)
nothing # hide

# ## State the targets
#
# The discrepancy scale is one unit for each quartile. Equal importance expresses
# an explicit preference for treating those two discrepancies equally during fitting.
# Five degrees of freedom is a family choice, not a quantity learned from two quartiles.

beliefs = (
    Approximate(:lower_quartile, Quantile(0.25), -0.5; scale=1.0, importance=1.0),
    Approximate(:upper_quartile, Quantile(0.75), 1.5; scale=1.0, importance=1.0),
)
nothing # hide

# ## Fit and inspect the candidates
#
# NelderMead needs no derivatives. Each family keeps the solution from each declared start.
# A solver result records termination, while validation records the achieved beliefs.

problem = Elicitation(beliefs, candidates)
# NelderMead is deterministic. Stochastic solvers need an explicitly configured backend RNG.
report = fit(Xoshiro(42), problem, NelderMead(); maxiters=2000, g_tol=1e-12)

checks = validate(Xoshiro(43), report)
comparison = compare(report)
nothing # hide

# Inspect the quartiles and a tail probability before selecting a family:

summary = map(report.results) do result
    prior = result.measure
    (;
        id=result.id,
        quartiles=map(
            p -> round(evaluate(Quantile(p), prior).value; digits=3),
            (0.25, 0.75),
        ),
        upper_tail=round(evaluate(Probability(4, Inf), prior).value; digits=4),
    )
end
display(collect(summary))

# The fitted quartiles are near the targets in both families. The Student-t tail is heavier.
# Two quartiles do not identify tail behavior, so use domain knowledge and further beliefs
# to assess that difference. `compare(report)` compares discrepancies for each target,
# retaining tradeoffs and ties. Small numerical differences do not settle the family choice.
# These retained points trade off the two target discrepancies:

display(comparison.incomparable)
#
# ## Select and use a prior
#
# Approximate targets do not impose hard tolerances. Review their achieved values before
# acceptance, or add `Hard` beliefs when a tolerance is required.

# Selection is an explicit analyst decision, separate from fit quality.
selection = select(
    report,
    (:normal, 1);
    validation=checks,
    rationale="Use the Gaussian candidate for this demonstration after reviewing both families.",
)
prior = measure(selection)
draws = rand(Xoshiro(44), prior^10)
display(draws)
