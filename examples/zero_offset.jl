# # [Check a calibration tolerance](@id zero-offset-tutorial)
#
# A balance needs a prior for its residual zero offset in milligrams. Calibration
# requires at least 90% probability within 2 mg of zero. Compare two fixed candidate
# laws centered at zero, both with a scale parameter of 1 mg.

using Credence, MeasureBase, Random
using CairoMakie

beliefs = (:offset => (Hard(:tolerance, Probability(-2, 2), AtLeast(0.9)),),)
candidates = (
    :offset => (
        Candidate(:normal, Returns(StdNormal()), (NamedTuple(),)),
        Candidate(:logistic, Returns(StdLogistic()), (NamedTuple(),)),
    ),
)
problem = Elicitation(beliefs, candidates)
report = fit(Xoshiro(101), problem)
validation = validate(Xoshiro(102), report)
belief_summary(validation)

# ## Compare the fitted priors
#
# The shaded band is the calibration tolerance. The Logistic puts less mass near
# zero. Vary the tolerance to see how wide a band each prior needs to reach 90%.

mkpath("elicitation-output")
offset_density = priorplot(validation, :offset; bounds=(-6, 6), xlabel="Zero offset (mg)")
vspan!(content(offset_density[1, 1]), -2, 2; color=(:gray, 0.15))
save("elicitation-output/offset-density.svg", offset_density)

# ![Candidate densities and the calibration tolerance.](elicitation-output/offset-density.svg)

tolerance = Figure(; size=(660, 380))
axis = Axis(
    tolerance[1, 1];
    xlabel="Tolerance half-width (mg)",
    ylabel="Probability within tolerance",
)
widths = range(0.1, 5; length=150)
for (result, label) in zip(report.results, ("Normal", "Logistic"))
    values = [evaluate(Probability(-w, w), result.measure).value for w in widths]
    lines!(axis, widths, values; label)
end
hlines!(axis, [0.9]; color=:gray, linestyle=:dash, label="Required coverage")
vlines!(axis, [2]; color=:black, linestyle=:dot, label="Current tolerance")
ylims!(axis, 0, 1.02)
axislegend(axis; position=:rb)
save("elicitation-output/offset-tolerance.svg", tolerance)

# ![Coverage across possible calibration tolerances.](elicitation-output/offset-tolerance.svg)
#
# The Normal assigns about 95% probability to the tolerance band. The Logistic
# assigns about 76%. Select the candidate that meets the requirement and record why.

choice = select(
    report,
    :offset,
    (:normal, 1);
    validation,
    rationale="The Normal meets the zero-offset tolerance.",
)
prior = measure(choice)
rand(Xoshiro(103), prior^4)

# `prior^4` is the product of four copies of the prior. Sampling from it gives
# four independent offsets.
