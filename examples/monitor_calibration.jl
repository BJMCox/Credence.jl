# # [Revise and save a calibration prior](@id monitor-calibration-tutorial)
#
# An air-quality monitor has a positive response lag in seconds and a signed
# concentration offset in μg/m³. Calibration staff specify tolerance requirements
# and estimate a mean and standard deviation for each parameter.

using Credence, Distributions, MeasureBase, Optimization, OptimizationOptimJL, Random
import ForwardDiff

beliefs = (
    :lag => (
        Hard(:responds_by_25s, Probability(0, 25), AtLeast(0.95)),
        Hard(:takes_over_3s, Probability(3, Inf), AtLeast(0.95)),
        Approximate(:mean, Mean(), 10.0; scale=2.0),
        Approximate(:spread, Std(), 4.0; scale=1.0),
    ),
    :offset => (
        Hard(:mean_bias, Mean(), Between(-0.5, 0.5)),
        Hard(:within_3, Probability(-3, 3), AtLeast(0.95)),
        Approximate(:mean, Mean(), -0.2; scale=0.15),
        Approximate(:spread, Std(), 1.1; scale=0.25),
    ),
)
candidates = (
    :lag => (
        :lognormal => LogNormalFamily(starts=(location=log(9.0), log_scale=log(0.4))),
        :gamma => GammaFamily(starts=(log_shape=log(6.0), log_scale=log(1.7))),
    ),
    :offset => (:normal => NormalFamily(starts=(location=-0.2, log_scale=0.0)),),
)
problem = Elicitation(
    beliefs,
    candidates;
    combinations=(
        :lognormal_lag => Independent(:lag=>:lognormal, :offset=>:normal),
        :gamma_lag => Independent(:lag=>:gamma, :offset=>:normal),
    ),
)

# Compare the two independent combinations. IPNewton uses the hard beliefs
# as constraints and minimizes the discrepancies from the approximate targets.

options = (; adtype=Optimization.AutoForwardDiff(), maxiters=1000)
report = fit(Xoshiro(301), problem, IPNewton(); options...)
validation = validate(Xoshiro(302), report)
compare(report; validation)

# The staff revise their lag estimate to 11 seconds after reviewing operation
# at a lower flow rate. Replace that one belief and validate the new fit.

revised = replace(problem, :lag => Approximate(:mean, Mean(), 11.0; scale=2.0))
refitted = fit(Xoshiro(303), revised, IPNewton(); options...)
checks = validate(Xoshiro(304), refitted)
belief_summary(checks; result_id=(:lognormal_lag, 1))

# Save the accepted prior with its fit and validation. Keep the original problem
# and the reason for the revision in the session history.

choice = select(
    refitted,
    (:lognormal_lag, 1);
    validation=checks,
    rationale="The LogNormal models multiplicative variation in response lag.",
)
restored = mktempdir() do directory
    path = joinpath(directory, "monitor.toml")
    save_session(
        path,
        revised;
        report=refitted,
        validation=checks,
        selections=(choice,),
        history=((; problem, reason="Revised the lag estimate for lower flow."),),
    )
    load_session(path)
end
rand(Xoshiro(305), measure(only(restored.selections))^4)
