# # [Check a calibration tolerance](@id zero-offset-tutorial)
#
# A balance needs a prior for its residual zero offset in milligrams. Calibration
# requires at least 90% probability within 2 mg of zero. Compare two fixed candidate
# laws centered at zero, both with a scale parameter of 1 mg.

using Credence, MeasureBase, Random

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
