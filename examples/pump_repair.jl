# # [Compare families for a repair time](@id pump-repair-tutorial)
#
# A maintenance engineer estimates a median of 2 hours for a pump repair and
# expects 90% of repairs to finish within 6 hours. Fit both a LogNormal and a Weibull.

using Credence, Distributions, Optimization, OptimizationOptimJL, Random
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
