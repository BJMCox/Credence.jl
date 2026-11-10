# # [Programmatic elicitation](@id scalar-tutorial)
#
# The unknown quantity is a sensor's fixed mean offset from a reference at 20°C,
# in degrees Celsius. The prior describes uncertainty about that offset, not reading noise.
# An expert supplies quartiles, then an analyst records a family choice after reviewing feedback.
# This script replays those inputs. The [guided REPL session](@ref guided-tutorial)
# shows how the expert arrives at the same final answers through discussion and revision.
#
# Run this file from the package directory with Julia 1.13 or later:
#
# ```sh
# julia --project=examples -e 'using Pkg; Pkg.instantiate()'
# julia --project=examples examples/scalar.jl
# ```
#
# The script writes its plot to `elicitation-output/prior-feedback.svg` in the working directory.
#
# ## Define the families
#
using Credence
using Distributions: LocationScale, Normal, TDist
using MeasureBase
using Optimization
using OptimizationOptimJL
using Random: Xoshiro
include(joinpath(pkgdir(Credence), "examples", "feedback.jl"))
nothing # hide

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
# These recorded answers concern the unknown offset: 25% prior probability lies at or below
# `q25`, and 75% lies at or below `q75`. They are illustrative judgments, not neutral defaults.
# Do not substitute a number when an expert cannot judge.
# The discrepancy scale is one degree Celsius for each quartile. Equal importance expresses
# an explicit preference for treating those two discrepancies equally during fitting.
# Five degrees of freedom is a family choice, not a quantity learned from two quartiles.

answers = (q25=-0.5, q75=1.0)
beliefs = (
    Approximate(:lower_quartile, Quantile(0.25), answers.q25; scale=1.0, importance=1.0),
    Approximate(:upper_quartile, Quantile(0.75), answers.q75; scale=1.0, importance=1.0),
)
nothing # hide

# ## Fit and inspect the candidates
#
# NelderMead needs no derivatives. Each family keeps the solution from each declared start.
# A solver result records termination, while validation records the achieved beliefs.

problem = Elicitation(beliefs, candidates)
# NelderMead is deterministic. Stochastic solvers need an explicitly configured backend RNG.
report = fit(Xoshiro(42), problem, NelderMead(); maxiters=2000, g_tol=1e-12)

comparison = compare(report)
nothing # hide

# Check every start's error before reading its measure, then inspect solver termination
# alongside the achieved values below.
# Failed starts remain in the report. Solver success does not certify the expert's beliefs.

display([(; result.id, result.error) for result in report.results])
all(result -> result.error === nothing, report.results) ||
    error("Inspect the failed starts before continuing this example.")

# Inspect the achieved quartiles and an unelicited tail probability before selecting a family.
# Rounding affects only the display:

summary = map(report.results) do result
    prior = result.measure
    (;
        id=result.id,
        quartiles=map(
            p -> round(evaluate(Quantile(p), prior).value; digits=3),
            (0.25, 0.75),
        ),
        upper_tail=round(evaluate(Probability(4, Inf), prior).value; digits=4),
        solver=string(result.solver.retcode),
    )
end
display(collect(summary))

# ## Review graphical feedback
#
# The shared plotting helper uses the fitted native measures. Each curve is a retained fit,
# not a mixture component or a probability weight on a family. The plot window is not a support bound.

figure = feedback(report)
mkpath("elicitation-output")
save("elicitation-output/prior-feedback.svg", figure);
println("Wrote elicitation-output/prior-feedback.svg") #!md
nothing # hide

# ![Prior density and cumulative probability for each retained fit.](elicitation-output/prior-feedback.svg)

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
# Fresh validation records the actual statistics. Here each check is `:target`, because
# these two judgments are approximate targets rather than hard requirements.

checks = validate(Xoshiro(43), report)
display([[check.status for check in row] for row in checks.checks])

# The decision is another recorded input. It is not inferred from the smallest residual
# or from the order of the candidates. Keep it with the questions, answers, script, seeds,
# and pinned environment when replaying this elicitation.

decision = (
    id=(:normal, 1),
    rationale="The expert accepts the Normal family after reviewing both families and their tails.",
)
selection = select(report, decision.id; validation=checks, rationale=decision.rationale)
prior = measure(selection)
draws = rand(Xoshiro(44), prior^10)
display(draws)

# `prior` is the selected native MeasureBase measure, in the offset's original units.
# Loading this file in a REPL also leaves `report`, `checks`, `selection`, and `figure`
# available for inspection. The [Wren tutorial](@ref wren-tutorial) shows model integration.
