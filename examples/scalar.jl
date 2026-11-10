# # [Programmatic signal-prior elicitation](@id scalar-tutorial)
#
# Fit priors for a weak Gaussian line above an exponential background in five detectors.
# The [model definition](@ref rare-event-model) contains the full Julia implementation,
# units, support bounds, and hierarchical background model.
# The [guided REPL session](@ref guided-tutorial) arrives at these same final judgments.
#
# Run from the package directory with Julia 1.13 or later:
#
# ```sh
# julia --project=examples -e 'using Pkg; Pkg.instantiate()'
# julia --project=examples examples/scalar.jl
# ```
#
# Plots are written to `elicitation-output` in the working directory.

using Credence, MeasureBase, Optimization, OptimizationOptimJL, Random
include(joinpath(pkgdir(Credence), "examples", "signal_background.jl"))
include(joinpath(pkgdir(Credence), "examples", "feedback.jl"))
nothing # hide

# ## Record the external judgments
#
# Elicit six hyperparameters: signal rate `S`, line location `mu`, signal width `width`,
# background log variation `sigma_B`, mean background rate `m_B`, and energy scale `lam`.
# `width` describes individual signal energies around `mu`. Its prior describes uncertainty
# in that detector resolution. Every candidate enforces the bounds in `prior_bounds`.
# These hard bounds differ from approximate targets for 95% prior upper quantiles.
# The location families also assume symmetry about 125 keV.
#
# These illustrative answers come from external knowledge, independently of search events.
# Each positive quantity has a median and a 95% upper quantile. The location has 90%
# probability between 124 and 126 keV. Rates use events/year and energy quantities use keV.

beliefs = (
    :S => (
        Approximate(:median, Quantile(0.5), 1.2; scale=1.0),
        Approximate(:upper, Quantile(0.95), 3.5; scale=1.0),
    ),
    :mu => (Approximate(:central, Probability(location_interval...), 0.9; scale=0.1),),
    :width => (
        Approximate(:median, Quantile(0.5), 1.6; scale=0.25),
        Approximate(:upper, Quantile(0.95), 2.1; scale=0.25),
    ),
    :sigma_B => (
        Approximate(:median, Quantile(0.5), 0.4; scale=0.1),
        Approximate(:upper, Quantile(0.95), 0.6; scale=0.1),
    ),
    :m_B => (
        Approximate(:median, Quantile(0.5), 7.0; scale=2.0),
        Approximate(:upper, Quantile(0.95), 10.5; scale=2.0),
    ),
    :lam => (
        Approximate(:median, Quantile(0.5), 55.0; scale=15.0),
        Approximate(:upper, Quantile(0.95), 80.0; scale=15.0),
    ),
)
problem = Elicitation(beliefs, signal_candidates(); combinations=signal_combinations())
nothing # hide

# ## Fit and compare candidate hyperpriors
#
# The candidates compare bounded Beta and LogNormal shapes for the five positive
# quantities, and Normal and Beta location shapes. All six hyperparameters are independent
# within each candidate. The five backgrounds share uncertain hyperparameters.

report = fit(Xoshiro(42), problem, NelderMead(); maxiters=20000, g_tol=1e-12)
display(collect(prior_summary(report)))
all(result -> result.error === nothing, report.results) ||
    error("Inspect failed starts before continuing.")
display(collect(target_summary(report, (:beta_normal, 1))))
comparison = compare(report)
display(comparison.incomparable)

# Solver codes and achieved targets are retained for every start. Pareto comparisons
# describe agreement with the targets. Review the family assumptions when selecting a prior.
#
# ## Inspect all six marginal priors
#
# Each curve represents one retained fit. Endpoints are omitted because a continuous Beta
# density can diverge at a boundary without assigning a point mass there.

mkpath("elicitation-output")
strength_figure = feedback(
    report,
    :S,
    strength_bounds;
    xlabel="Signal rate (events/year)",
    levels=(0.5, 0.95),
)
save("elicitation-output/signal-prior.svg", strength_figure);
marginal_figure = hyperprior_feedback(report)
save("elicitation-output/hyperpriors.svg", marginal_figure);
nothing # hide

# ![Candidate signal-rate densities and cumulative probabilities.](elicitation-output/signal-prior.svg)
# ![Cumulative probabilities for all six elicited hyperparameters.](elicitation-output/hyperpriors.svg)
#
# The 95% quantile target is not a posterior upper limit. Continuous strength priors
# assign no probability mass to exactly `S = 0`; a background-only model comparison
# would require a separate null hypothesis.
#
# ## Validate and select explicitly
#
# Approximate targets receive the status `:target`. Review their achieved values when
# selecting a prior. Physical support restrictions come from the bounded candidate laws.

validation = validate(Xoshiro(43), report)
display([[check.status for check in row] for row in validation.checks])
decision = (
    id=(:beta_normal, 1),
    rationale="The expert accepts the bounded Beta hyperparameters and calibrated Normal location after reviewing both alternatives.",
)
selection = select(report, decision.id; validation, rationale=decision.rationale)
prior = measure(selection)
nothing # hide

# ## Use the prior in the scientific model
#
# The selected object is a native hyperprior. Draw the five background rates conditionally
# to obtain the full hierarchical prior. The lognormal correction preserves their mean `m_B`.
# Fixed generating parameters make the following synthetic dataset reproducible.
# Elicitation uses the external judgments recorded above.

truth = (S=0.8, mu=125.4, width=1.6, sigma_B=0.35, m_B=7.0, lam=55.0)
truth_parameters = background_parameters(Xoshiro(2026), truth)
data = simulate_spectrum(Xoshiro(2027), truth_parameters)
display((
    exposure=experiment.exposure,
    efficiency=experiment.efficiency,
    counts=data.counts,
    events=length(data.energy),
))

rng = Xoshiro(44)
hyperparameters = rand(rng, prior)
parameters = background_parameters(rng, hyperparameters)
display((
    S=parameters.S,
    mu=parameters.mu,
    width=parameters.width,
    B=parameters.B,
    loglikelihood=spectrum_loglikelihood(parameters, data),
))

# This is an eleven-dimensional parameter point: six hyperparameters and five backgrounds.
# Inspect paired draws to retain their joint meaning. The left panel reflects the assumed
# hyperparameter independence. Its colour shows uncertain signal width. The right panel
# shows dependence induced by the shared uncertain background mean and variation.

joint_figure = joint_feedback(Xoshiro(46), prior)
save("elicitation-output/joint-prior.svg", joint_figure);
nothing # hide

# ![Joint signal and hierarchical background prior draws.](elicitation-output/joint-prior.svg)
#
# The likelihood combines per-detector counts with individual event energies.
# Julia's `Exponential(lam)` uses `lam` as its mean scale in keV.
# The likelihood includes every event energy. Binning serves only the display.
#
# ## Inspect prior predictive implications
#
# Use 10,000 hierarchical prior predictive draws. Blue curves show expected binned spectra
# for the first 200 draws. Black points are the synthetic observed counts per bin.
# The right panel uses all 10,000 Poisson count draws, with the observed total marked in black.
# These prior predictions describe the selected prior's implications for observable data.
# Feedback used to revise elicited judgments must remain independent of these search observations.

predictive_figure = spectrum_feedback(Xoshiro(45), prior, data; ndraws=10_000)
save("elicitation-output/prior-predictive.svg", predictive_figure);
println("Plots saved in elicitation-output") #!md
nothing # hide

# ![Prior expected spectra and predictive total counts, compared with synthetic observations.](elicitation-output/prior-predictive.svg)
