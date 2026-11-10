using Credence
using Distributions: LocationScale, Normal, TDist
using MeasureBase
using Optimization
using OptimizationOptimJL
using Random

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
beliefs = (
    Approximate(:lower_quartile, Quantile(0.25), -0.5; scale=1.0, importance=1.0),
    Approximate(:upper_quartile, Quantile(0.75), 1.5; scale=1.0, importance=1.0),
)

problem = Elicitation(beliefs, candidates)
# NelderMead is deterministic. Stochastic solvers need an explicitly configured backend RNG.
report = fit(Xoshiro(42), problem, NelderMead(); maxiters=2000, g_tol=1e-12)

checks = validate(Xoshiro(43), report)
comparison = compare(report)

# Selection is an explicit analyst decision, separate from fit quality.
selection = select(
    report,
    (:normal, 1);
    validation=checks,
    rationale="Use the Gaussian candidate for this demonstration after reviewing both families.",
)
prior = measure(selection)
draws = rand(Xoshiro(44), prior^10)
