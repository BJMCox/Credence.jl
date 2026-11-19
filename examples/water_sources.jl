# # [Elicit water-source fractions](@id water-sources-tutorial)
#
# A treatment plant receives water from a reservoir, a river, and wells.
# The three source fractions sum to one. Staff give bounds on expected fractions
# for the reservoir and the wells, and estimate the spread of the reservoir fraction.

using Credence, Distributions, Optimization, OptimizationOptimJL, Random
using CairoMakie
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
validation

# The fitted means can be compared with the elicited targets and bounds. The
# river mean follows from the sum-to-one constraint.

mkpath("elicitation-output")
source_names = ["Reservoir", "River", "Wells"]
summary = belief_summary(validation)
mean_bounds = only(row for row in summary if last(row.belief_id) == :mean_bounds)
mean_targets = only(row for row in summary if last(row.belief_id) == :means)
lower, upper = mean_bounds.target
means = Figure(; size=(660, 360))
axis = Axis(means[1, 1]; ylabel="Mean source fraction", xticks=(1:3, source_names))
rangebars!(
    axis,
    [1, 3],
    Float64[lower[1], 0],
    Float64[upper[1], upper[3]];
    color=:gray,
    linewidth=3,
    whiskerwidth=14,
    label="Allowed mean range",
)
scatter!(
    axis,
    [1, 3],
    Float64[mean_targets.target[1], mean_targets.target[3]];
    marker=:diamond,
    color=:darkorange,
    markersize=14,
    label="Elicited target",
)
scatter!(axis, 1:3, mean_bounds.value; color=:steelblue, label="Fitted mean")
xlims!(axis, 0.6, 3.4)
ylims!(axis, -0.02, 0.68)
axislegend(axis; position=:rt)
save("elicitation-output/water-means.svg", means)

# ![Fitted source means, elicited targets, and allowed ranges.](elicitation-output/water-means.svg)
#
# Review the implied means and spread before selecting the joint prior.

choice = select(
    report,
    :shares,
    (:dirichlet, 1);
    validation,
    rationale="The fitted fractions satisfy the mean bounds and match the reservoir spread.",
)
rand(Xoshiro(403), measure(choice)^4)

# The triangle shows 1,000 draws from the joint prior. Each vertex represents
# all flow coming from one source. Moving toward a vertex increases its fraction.

draws = rand(Xoshiro(404), measure(choice)^1000)
height = sqrt(3) / 2
simplex = Figure(; size=(490, 420))
axis = Axis(simplex[1, 1]; aspect=DataAspect())
lines!(axis, [0, 1, 0.5, 0], [0, 0, height, 0]; color=:gray)
scatter!(
    axis,
    [p[2]+p[3]/2 for p in draws],
    [height*p[3] for p in draws];
    color=(:steelblue, 0.35),
    markersize=4,
)
text!(
    axis,
    [0, 1, 0.5],
    [-0.045, -0.045, height+0.04];
    text=source_names,
    align=(:center, :center),
)
hidedecorations!(axis)
hidespines!(axis)
xlims!(axis, -0.15, 1.15)
ylims!(axis, -0.08, height+0.09)
save("elicitation-output/water-simplex.svg", simplex)

# ![Dirichlet draws on the three-source simplex.](elicitation-output/water-simplex.svg)
