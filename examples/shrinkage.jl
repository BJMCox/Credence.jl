# # [Compare shrinkage priors for building load](@id shrinkage-tutorial)
#
# A building-load model uses two weather predictors and two occupancy predictors.
# Predictors and load are standardized. The expert expects modest load variation
# for a specified operating condition and allows for occasional large coefficients.

using Credence, Distributions, MeasureBase, Random
using CairoMakie

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
validation

# Compare the tails of the largest absolute coefficient using 10,000 prior draws
# from each family. The reference point marks the elicited 15% chance of exceeding 1.

mkpath("elicitation-output")
tail = Figure(; size=(660, 380))
axis = Axis(
    tail[1, 1];
    xlabel="Largest absolute coefficient",
    ylabel="Probability of exceeding threshold",
)
thresholds = range(0, 3; length=200)
rng = Xoshiro(504)
labels = ("Regularized horseshoe", "Group horseshoe")
for (result, label) in zip(report.results, labels)
    largest =
        sort!([maximum(abs, rand(rng, result.measure).coefficients) for _ in 1:10_000])
    exceedance = [1 - searchsortedlast(largest, t)/length(largest) for t in thresholds]
    lines!(axis, thresholds, exceedance; label)
end
scatter!(axis, [1], [0.15]; marker=:diamond, color=:black, label="Elicited probability")
axislegend(axis)
save("elicitation-output/shrinkage-tails.svg", tail)

# ![Tail probabilities for the largest coefficient under each shrinkage prior.](elicitation-output/shrinkage-tails.svg)
#
# The load plot compares the independently validated probabilities and their
# Monte Carlo enclosures against the central target and the wider hard minimum.

load_checks = Figure(; size=(700, 360))
axis = Axis(
    load_checks[1, 1];
    ylabel="Prior predictive probability",
    xticks=(1:2, ["Load within ±0.7", "Load within ±2"]),
)
for (i, (result, label)) in enumerate(zip(report.results, labels))
    rows = belief_summary(validation; result_id=result.id)
    selected = [
        only(row for row in rows if last(row.belief_id) == id) for
        id in (:central_load, :load_band)
    ]
    x = (1:2) .+ 0.18 * (i - 1.5)
    y = getproperty.(selected, :value)
    color = (:steelblue, :darkorange)[i]
    errorbars!(
        axis,
        x,
        y,
        y .- getproperty.(selected, :lower),
        getproperty.(selected, :upper) .- y;
        color,
        whiskerwidth=10,
    )
    scatter!(axis, x, y; color, label)
end
scatter!(axis, [1], [0.75]; marker=:diamond, color=:black, label="Approximate target")
lines!(
    axis,
    [1.7, 2.3],
    [0.85, 0.85];
    color=:black,
    linestyle=:dash,
    label="Required minimum",
)
xlims!(axis, 0.6, 2.4)
ylims!(axis, 0.45, 1.02)
axislegend(axis; position=:lb)
save("elicitation-output/shrinkage-load.svg", load_checks)

# ![Predictive probability checks with Monte Carlo enclosures.](elicitation-output/shrinkage-load.svg)
#
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
