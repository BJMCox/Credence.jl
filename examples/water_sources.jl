# # [Elicit water-source fractions](@id water-sources-tutorial)
#
# A treatment plant receives water from a reservoir, a river, and wells.
# The three source fractions sum to one. Staff give bounds on expected fractions
# for the reservoir and the wells, and estimate the spread of the reservoir fraction.

using Credence, Distributions, Optimization, OptimizationOptimJL, Random
using VarNames: @vn
import ForwardDiff

beliefs = (
    :shares => (
        Hard(:mean_bounds, Mean(), Between([0.5, nothing, nothing], [0.62, nothing, 0.19])),
        Approximate(:means, Mean(), [0.55, nothing, 0.15]; scale=0.04),
    ),
    @vn(shares[1]) => (Approximate(:spread, Std(), 0.10; scale=0.04),),
)
candidates = (
    :shares =>
        (:dirichlet => DirichletFamily(3; starts=(log_alpha=log.([11.0, 6.0, 3.0]),)),),
)

# `nothing` leaves that element unspecified. The simplex fixes the river mean
# once the other means are fitted. The target for the reservoir's standard
# deviation informs the Dirichlet concentration.

problem = Elicitation(beliefs, candidates)
report = fit(
    Xoshiro(401),
    problem,
    IPNewton();
    adtype=Optimization.AutoForwardDiff(),
    maxiters=1000,
)
validation = validate(Xoshiro(402), report)
belief_summary(validation)

# Review the implied means and spread before selecting the joint prior.

choice = select(
    report,
    :shares,
    (:dirichlet, 1);
    validation,
    rationale="The fitted fractions satisfy the mean bounds and match the reservoir spread.",
)
rand(Xoshiro(403), measure(choice)^4)
