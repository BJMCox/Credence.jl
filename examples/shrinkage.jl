# # [Compare shrinkage priors for building load](@id shrinkage-tutorial)
#
# A building-load model uses two weather predictors and two occupancy predictors.
# Predictors and load are standardized. The expert expects modest load variation
# for a specified operating condition and allows for occasional large coefficients.

using Credence, Distributions, MeasureBase, Random

groups = (:weather, :weather, :occupancy, :occupancy)
candidates = (
    :effects => (
        :regularized =>
            RegularizedHorseshoeFamily(4; scale=0.12, slab_scale=0.7, slab_df=4),
        :grouped => GroupHorseshoeFamily(groups; scale=0.08),
    ),
)

# The first prior limits the spread of large coefficients through a slab. The
# second shares a shrinkage scale within each predictor group. Both return a
# record of coefficients and latent scales.
#
# Assess the largest absolute coefficient and the predicted load at one set of
# predictor values. The observation model has a fixed residual deviation of 0.25.

predictors = (1.0, 0.5, -0.5, 0.75)
function observation(effects)
    location = sum(effects.coefficients .* predictors)
    return asmeasure(Normal(location, 0.25))
end

largest_effect =
    Derived(:largest_effect, :effects, effects -> maximum(abs, effects.coefficients))
future_load = Predictive(:future_load, :effects, observation)
beliefs = (
    largest_effect =>
        (Approximate(:large_effect, Probability(1, Inf), 0.15; scale=0.08),),
    future_load => (
        Hard(:load_band, Probability(-2, 2), AtLeast(0.85)),
        Approximate(:central_load, Probability(-0.7, 0.7), 0.75; scale=0.08),
    ),
)

# Evaluate both fixed priors by Monte Carlo. Use independent draws for validation,
# and inspect the uncertainty bounds for the hard requirement.

problem = Elicitation(beliefs, candidates)
report = fit(Xoshiro(501), problem; sampling=MonteCarlo(10_000))
validation = validate(Xoshiro(502), report)
map(belief_summary(validation)) do row
    (;
        family=row.result_id[2],
        belief=last(row.belief_id),
        row.value,
        row.lower,
        row.upper,
        row.status,
    )
end

# Select the regularized prior after checking its predictions. Draw the
# coefficients and their scales together, then draw a load given those values.

choice = select(
    report,
    :effects,
    (:regularized, 1);
    validation,
    rationale="The load band is supported and the slab limits the coefficient spread.",
)
rng = Xoshiro(503)
effects = rand(rng, measure(choice))
(; effects, load=rand(rng, observation(effects)))
