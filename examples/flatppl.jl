# # [Using an elicited prior in FlatPPL](@id flatppl-tutorial)
#
# A detector records events with uncertain efficiency. Its detected count over a fixed
# exposure follows a Poisson law with mean `exposure * efficiency * rate`.
# We elicit a joint prior for efficiency and event rate in Julia, export that prior,
# and use it in a FlatPPL observation model.
#
# Run the example from the package directory with Julia 1.13 or later:
#
# ```sh
# julia --project=examples -e 'using Pkg; Pkg.instantiate()'
# julia --project=examples examples/flatppl.jl
# ```

using Credence, MeasureBase, Optimization, OptimizationOptimJL, Random
using Distributions: Poisson, logpdf

# ## Elicit efficiency and event rate
#
# Suppose calibration knowledge gives efficiency median 0.8 and fifth percentile 0.6.
# The event rate has median 4 events/hour and 95th percentile 9 events/hour.
# Scales set the units of fitting discrepancies. We assume a Beta efficiency prior,
# a Gamma rate prior, and independence between the two parameters.

beliefs = (
    :efficiency => (
        Approximate(:median, Quantile(0.5), 0.8; scale=0.1),
        Approximate(:lower, Quantile(0.05), 0.6; scale=0.1),
    ),
    :rate => (
        Approximate(:median, Quantile(0.5), 4.0; scale=2.0),
        Approximate(:upper, Quantile(0.95), 9.0; scale=2.0),
    ),
)
candidates = (
    :efficiency => (:beta => ScaledBetaFamily(0.0, 1.0),),
    :rate => (:gamma => GammaFamily(),),
)
problem = Elicitation(
    beliefs,
    candidates;
    combinations=(:calibration => Independent(:efficiency => :beta, :rate => :gamma),),
)
report = fit(Xoshiro(72), problem, NelderMead(); maxiters=10000, g_tol=1e-12)
validation = validate(Xoshiro(73), report)
selected_id = (:calibration, 1)
display(belief_summary(validation; result_id=selected_id))

# Review the fitted quantiles and record acceptance of the family assumptions.

selection = select(
    report,
    selected_id;
    validation,
    rationale="The fitted quantiles represent the calibration judgments under independent Beta and Gamma priors.",
)
prior = measure(selection)
nothing # hide

# ## Export the prior and define the likelihood
#
# `flatppl(selection)` returns a formatted measure expression. Its named factors retain
# the fitted parameters and support. Insert that expression into a FlatPPL source file.
#
# The model below uses three hours of exposure and an observed count of 11.
# `elementof` declares each parameter's domain. `kernelof` constructs the conditional
# count distribution, and `likelihoodof` fixes the observed count.
# `bayesupdate` multiplies this likelihood by the exported prior to form an unnormalized posterior.

source = """
flatppl_compat = "0.1"

prior = $(flatppl(selection))

exposure = 3.0
observed_count = 11

efficiency = elementof(interval(0.0, 1.0))
rate = elementof(posreals)
count ~ Poisson(exposure * efficiency * rate)
observation_model = kernelof(count, efficiency = efficiency, rate = rate)
likelihood = likelihoodof(observation_model, observed_count)
posterior = bayesupdate(likelihood, prior)

point = record(efficiency = 0.8, rate = 4.0)
prior_logdensity = logdensityof(prior, point)
posterior_logdensity = logdensityof(posterior, point)
"""
println(source)
mkpath("elicitation-output")
write("elicitation-output/counting-model.flatppl", source)
nothing # hide

# [Download the complete FlatPPL model](elicitation-output/counting-model.flatppl).
# Open this file in a FlatPPL environment to inspect the prior, likelihood, and posterior.
# The fixed `point` also provides a density check against Julia.
#
# ## Check the density in Julia
#
# At the same parameter point, the posterior log density is the sum of the prior
# log density and Poisson log likelihood. The FlatPPL bindings `prior_logdensity` and
# `posterior_logdensity` should match the values below.

point = (efficiency=0.8, rate=4.0)
prior_logdensity = logdensityof(prior, point)
posterior_logdensity =
    prior_logdensity + logpdf(Poisson(3.0 * point.efficiency * point.rate), 11)
display((; point, prior_logdensity, posterior_logdensity))

# The posterior measure includes the likelihood but retains its unknown normalizing constant.
# Posterior inference can use this density in the host environment.
