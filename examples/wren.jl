# Run with the integration environment:
# julia --project=test/integration examples/wren.jl
using Credence
using MeasureBase: MeasureBase
using Distributions: Normal, MvNormal, LogNormal, mean
using Plaice
using PDMats
using Optimization, OptimizationOptimJL
using Wren: @wren, @observe, nuts, AutoEnzyme, Reverse
import WrenCore
import Enzyme
using Random: Xoshiro

covariance = [1.0 0.4; 0.4 0.8]
function build_prior(p)
    MeasureBase.productmeasure((
        beta=MeasureBase.asmeasure(MvNormal(p[1:2], covariance)),
        sigma=MeasureBase.asmeasure(LogNormal(p[3], 0.4)),
    ))
end
candidate = Candidate(:joint, build_prior, [0.0, 0.0, 0.0])

beliefs = (
    Approximate(
        :first_coefficient,
        Statistic(p -> mean(MeasureBase.marginals(p).beta)[1]),
        0.2;
        scale=1.0,
        importance=1.0,
    ),
    Approximate(
        :second_coefficient,
        Statistic(p -> mean(MeasureBase.marginals(p).beta)[2]),
        -0.1;
        scale=1.0,
        importance=1.0,
    ),
    Approximate(
        :log_scale,
        Statistic(p -> log(evaluate(Quantile(0.5), MeasureBase.marginals(p).sigma).value)),
        -0.3;
        scale=1.0,
        importance=1.0,
    ),
)
problem = Elicitation(beliefs, [candidate])
report = fit(Xoshiro(1001), problem, NelderMead(); maxiters=3000, g_tol=1e-14)
validation = validate(Xoshiro(1002), report)
selection = select(report, (:joint, 1); validation, rationale="Expert-selected joint prior")
prior = measure(selection)

@wren function regression(prior, y)
    theta ~ prior
    @observe y ~ Normal(theta.beta[1] + 2 * theta.beta[2], theta.sigma)
end

model = regression(prior, 0.4)
# This model only reads its fixed prior. Elide that copy alone to avoid Enzyme #3527.
# This uses the internal API of the pinned Wren version, not a general no-copy policy.
copies = filter(!=(:prior), WrenCore._copied_args(model))
model = WrenCore._with_copied_args(model, copies)
chain = nuts(
    Xoshiro(1003),
    model,
    200;
    adtype=AutoEnzyme(; mode=Reverse),
    reduce_copies=false,
    progress=false,
)
