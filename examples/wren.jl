# # [A joint prior in Wren](@id wren-tutorial)
#
# Fit a joint prior for two correlated coefficients and a positive observation scale.
# Pass the selected native measure into one Wren prior site, preserving coefficient dependence.
#
# Run from the package directory with Julia 1.13 or later:
#
# ```sh
# julia --project=test/integration -e 'using Pkg; Pkg.instantiate()'
# julia --project=test/integration examples/wren.jl
# ```
#
# Wren is currently private, so running this example requires repository access.
# Its environment pins Wren, Plaice, and MeasureBase separately from the core package.
#
# ## Define a joint family
#
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

# ## Fit the expert's targets
#
# Keep the coefficient covariance fixed. Fit their means and the log of the scale's median.
# A `Statistic` callback reads each property from the native product measure.

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

# ## Use one joint prior site
#
# Plaice maps the named product to unconstrained coordinates for Wren.
# The coefficient chart preserves the original coefficients. The positive scale uses a log chart.

@wren function regression(prior, y)
    theta ~ prior
    @observe y ~ Normal(theta.beta[1] + 2 * theta.beta[2], theta.sigma)
end

model = regression(prior, 0.4)

# ## Sample with NUTS
#
# This model reads its fixed prior without mutating it. Elide that copy alone to work around
# [Enzyme #3527](https://github.com/EnzymeAD/Enzyme.jl/issues/3527), related to
# [#3561](https://github.com/EnzymeAD/Enzyme.jl/issues/3561).
# This uses the pinned Wren version's internal API. Wren can mutate other model arguments,
# so copies of mutable work buffers must remain enabled.
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
display(size(chain))

# `chain` is ready for use as a FlexiChains chain. The 200 retained draws illustrate the
# interface; assess convergence and Monte Carlo error before using a posterior estimate.
