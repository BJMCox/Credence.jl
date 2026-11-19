# # [Compare families for a repair time](@id pump-repair-tutorial)
#
# A maintenance engineer estimates a median of 2 hours for a pump repair and
# expects 90% of repairs to finish within 6 hours. Fit both a LogNormal and a Weibull.

using Credence, Distributions, Optimization, OptimizationOptimJL, Random
using MeasureBase: logdensityof
using CairoMakie
import ForwardDiff

beliefs = (
    :hours => (
        Approximate(:median, Quantile(0.5), 2.0; scale=0.5),
        Approximate(:upper, Quantile(0.9), 6.0; scale=1.0),
    ),
)
candidates = (
    :hours => (
        :lognormal => LogNormalFamily(starts=(location=log(2.0), log_scale=log(0.8))),
        :weibull => WeibullFamily(starts=(log_shape=0.0, log_scale=log(3.0))),
    ),
)
problem = Elicitation(beliefs, candidates)
report = fit(Xoshiro(201), problem, BFGS(); adtype=Optimization.AutoForwardDiff())
validation = validate(Xoshiro(202), report)
belief_summary(validation)

# The scales express how much a discrepancy in hours contributes to fitting.
# Both families can match the two quantiles. Compare the implied chance of a repair
# lasting longer than an eight-hour shift before choosing a family.

map(report.results) do result
    (; id=result.id, exceeds_shift=evaluate(Probability(8, Inf), result.measure).value)
end

# The densities show where each family puts its mass around the common quantiles.
# The conditional completion rate shows how the chance of finishing in the next
# short interval changes as a repair remains open. It is the density divided by
# the probability of still being open.

mkpath("elicitation-output")
density = priorplot(report, :hours; bounds=(0, 14), xlabel="Repair time (hours)")
vlines!(content(density[1, 1]), [2, 6]; color=:gray, linestyle=:dash)
save("elicitation-output/repair-density.svg", density)

# ![Fitted repair duration densities with the elicited quantiles marked.](elicitation-output/repair-density.svg)

completion = Figure(; size=(660, 380))
axis = Axis(
    completion[1, 1];
    xlabel="Elapsed repair time (hours)",
    ylabel="Completion rate (per hour)",
)
hours = range(0.25, 16; length=200)
for (result, label) in zip(report.results, ("LogNormal", "Weibull"))
    rate = map(hours) do t
        exp(logdensityof(result.measure, t)) /
        evaluate(Probability(t, Inf), result.measure).value
    end
    lines!(axis, hours, rate; label)
end
vlines!(axis, [8]; color=:gray, linestyle=:dash, label="End of shift")
Legend(completion[2, 1], axis; orientation=:horizontal, framevisible=false)
save("elicitation-output/repair-completion.svg", completion)

# ![Conditional completion rate versus elapsed repair time.](elicitation-output/repair-completion.svg)
#
# Here the engineer chooses the LogNormal to represent multiplicative variation
# in the time needed to diagnose, access, and repair the pump.

choice = select(
    report,
    :hours,
    (:lognormal, 1);
    validation,
    rationale="The LogNormal represents multiplicative variation in repair duration.",
)
measure(choice)
