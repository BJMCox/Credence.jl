# # [Prior elicitation for a signal-background model](@id scalar-tutorial)
#
# A rare-event search distinguishes a narrow Gaussian signal from an exponential background.
# This example defines the model and fits priors to expert judgments about its parameters.
# It compares two families, revises a judgment, selects a prior, and examines prior predictions.
#
# Run the example from the package directory with Julia 1.13 or later:
#
# ```sh
# julia --project=examples -e 'using Pkg; Pkg.instantiate()'
# julia --project=examples examples/scalar.jl
# ```
#
# Figures are saved in `elicitation-output` in the working directory.

using Credence, MeasureBase, Optimization, OptimizationOptimJL, Random
include(joinpath(pkgdir(Credence), "examples", "signal_background.jl"))
include(joinpath(pkgdir(Credence), "examples", "feedback.jl"))
nothing # hide

# ## Elicit the hyperparameters
#
# The expert supplies judgments for signal rate `S`, line location `mu`, resolution `width`,
# background log standard deviation `sigma_B`, mean background rate `m_B`, and background
# energy scale `lam`. Resolution describes the spread of signal energies around the line location.
# Its prior describes uncertainty about that spread.
#
# For each positive parameter, elicit a median and 95% upper quantile. For location, elicit
# the probability of lying between 124 and 126 keV. Here that probability is 90%.
# The statements below represent information from calibration or earlier experiments,
# independent of the search observations. Their scales set the units of fitting discrepancies.

initial_beliefs = (
    :S => (
        Approximate(:median, Quantile(0.5), 1.2; scale=1.0),
        Approximate(:upper, Quantile(0.95), 5.0; scale=1.0),
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
initial_problem =
    Elicitation(initial_beliefs, signal_candidates(); combinations=signal_combinations())
nothing # hide

# ## Fit the candidate priors
#
# Fit both combinations to the judgments. The summaries show the largest scaled residual,
# solver outcome, and achieved statistics for each starting point.

initial_report =
    fit(Xoshiro(42), initial_problem, NelderMead(); maxiters=20000, g_tol=1e-12)
display(collect(prior_summary(initial_report)))
display(collect(target_summary(initial_report, (:beta_normal, 1))))

# ## Review a judgment
#
# The fitted priors imply probabilities for other ranges. For example, evaluate the
# probability that the signal rate lies near its upper bound, between 6 and 8 events/year.
# Discuss whether these probabilities agree with the expert's knowledge.

near_cap = [
    (result.id, evaluate(Probability(6.0, 8.0), marginals(result.measure).S).value) for
    result in initial_report.results
]
display(near_cap)
effective_exposure = sum(experiment.exposure .* experiment.efficiency)
display(initial_problem[:S, :upper].target * effective_exposure)

# Multiplying the rate quantile by effective exposure gives a quantile of expected signal
# yield. Realized signal counts also vary according to their Poisson law.
# Suppose this discussion leads the expert to revise the 95% rate quantile from 5 to
# 3.5 events/year. Record the reason, preserve the initial problem, and fit the revision.

revision_reason = "The expert revised the upper rate quantile after reviewing external rate knowledge and its expected-yield implication."
problem =
    replace(initial_problem, :S => Approximate(:upper, Quantile(0.95), 3.5; scale=1.0))
report = fit(Xoshiro(42), problem, NelderMead(); maxiters=20000, g_tol=1e-12)
display((original=initial_problem[:S, :upper].target, revised=problem[:S, :upper].target))
display(collect(prior_summary(report)))
all(result -> result.error === nothing, report.results) ||
    error("Inspect failed starts before continuing.")
display(collect(target_summary(report, (:beta_normal, 1))))
comparison = compare(report)
display(comparison.incomparable)

# Compare the revised fits by their discrepancies, then review the distributional assumptions.
#
# ## Plot the fitted priors
#
# Each curve represents a fitted candidate. The density plot excludes endpoints because
# a continuous Beta density may diverge there while assigning zero mass to the boundary.

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

# ![Signal-rate densities and cumulative probabilities for fitted candidates.](elicitation-output/signal-prior.svg)
# ![Fitted cumulative probabilities for the six hyperparameters.](elicitation-output/hyperpriors.svg)
#
# The upper quantile describes prior uncertainty about the signal rate. A posterior limit
# would also depend on the search data. These continuous priors give zero mass to exactly
# `S = 0`. Comparing a background-only hypothesis requires a separate null model.
#
# ## Validate and select a prior
#
# Validation evaluates the fitted judgments again. Approximate targets receive status
# `:target`, and the candidate supports enforce the parameter bounds. Review the achieved
# values and family assumptions before recording a selection.

validation = validate(Xoshiro(43), report)
display([[check.status for check in row] for row in validation.checks])
decision = (
    id=(:beta_normal, 1),
    rationale="The expert accepts the bounded Beta hyperparameters and calibrated Normal location after reviewing both alternatives.",
)
selection = select(report, decision.id; validation, rationale=decision.rationale)
prior = measure(selection)
nothing # hide

# ## Export the prior to FlatPPL
#
# `flatppl(selection)` produces a measure expression for the selected six-parameter
# hyperprior. The expression preserves the parameter names, distribution parameters,
# support bounds, and independence assumptions. Save it as a FlatPPL source file:

prior_expression = flatppl(selection)
write(
    "elicitation-output/signal-prior.flatppl",
    "flatppl_compat = \"0.1\"\n\nprior = " * prior_expression * "\n",
)
println(prior_expression)
nothing # hide

# [Download the exported hyperprior](elicitation-output/signal-prior.flatppl).
# The binding `prior` defines the joint measure in FlatPPL. Conditional background rates
# can then use its `m_B` and `sigma_B` parameters in the observation model.
# `flatppl(prior)` also accepts the MeasureBase measure directly.
#
# ## Generate data from the model
#
# The selected measure describes the six hyperparameters. Draw background rates conditional
# on their shared hyperparameters to obtain all eleven model parameters.
# Fixed generating values and seeds make the synthetic dataset below reproducible.

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

# Paired draws show the joint prior. The left panel displays signal rate and location,
# with signal width shown by colour. These hyperparameters are independent by assumption.
# The right panel displays two background rates. Their shared uncertain hyperparameters
# induce dependence. Their covariance is the prior variance of `m_B`.

joint_figure = joint_feedback(Xoshiro(46), prior)
save("elicitation-output/joint-prior.svg", joint_figure);
nothing # hide

# ![Paired draws from the signal and hierarchical background prior.](elicitation-output/joint-prior.svg)
#
# The likelihood evaluates the dataset counts and each event energy. `lam` is the mean
# scale of the exponential background in keV. Histogram bins serve only the figures.
#
# ## Examine prior predictions
#
# Generate 10,000 draws from the hierarchical prior and observation model. The left panel
# shows expected binned spectra for the first 200 draws, with synthetic observations in black.
# The right panel uses all 10,000 draws to show the distribution of total event counts.
# These plots describe how the chosen prior translates into observable quantities.
# Further prior revisions should use information independent of the search observations.

predictive_figure = spectrum_feedback(Xoshiro(45), prior, data; ndraws=10_000)
save("elicitation-output/prior-predictive.svg", predictive_figure);
println("Plots saved in elicitation-output") #!md
nothing # hide

# ![Expected spectra and total-count predictions under the selected prior.](elicitation-output/prior-predictive.svg)
