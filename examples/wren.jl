# # [Using an elicited prior in Wren](@id wren-tutorial)
#
# Consider a Normal regression with intercept `beta[1]`, slope `beta[2]`, and residual
# standard deviation `sigma`. We elicit their prior, then condition the model on one
# observation, `y = 0.4` at predictor value `x = 2`.
#
# The coefficients have a correlated Normal prior. The residual standard deviation has
# a LogNormal prior, independent of the coefficients. Credence fits the coefficient means
# and the median of `sigma`. Wren then uses the selected joint measure as its prior.
#
# Run the complete example from the package directory with Julia 1.13 or later:
#
# ```sh
# julia --project=test/integration -e 'using Pkg; Pkg.instantiate()'
# julia --project=test/integration examples/wren.jl
# ```
#
# This environment pins the development versions of Wren, Plaice, and MeasureBase.
# Installing Wren currently requires access to its private repository.

using Credence
using MeasureBase: MeasureBase
using Distributions: Normal, MvNormal, LogNormal
using Plaice
using PDMats
using Optimization, OptimizationOptimJL
using Wren: @wren, @observe, nuts, AutoEnzyme, Reverse
import WrenCore
import Enzyme
using Random: Xoshiro

# ## Specify the prior family
#
# Let `beta` have mean vector `p.beta` and covariance matrix `covariance`.
# Let `log(sigma)` have mean `p.log_sigma` and standard deviation 0.4.
# The covariance and log standard deviation are fixed family assumptions in this example.
# The product measure makes `beta` and `sigma` independent while preserving dependence
# between the two coefficients.

covariance = [1.0 0.4; 0.4 0.8]
function build_prior(p)
    MeasureBase.productmeasure((
        beta=MeasureBase.asmeasure(MvNormal(p.beta, covariance)),
        sigma=MeasureBase.asmeasure(LogNormal(p.log_sigma, 0.4)),
    ))
end
candidate = Candidate(:joint, build_prior, ((beta=zeros(2), log_sigma=0.0),))

# `Candidate` associates the family with its name and starting parameters. The builder
# receives the named parameter record and returns a normalized MeasureBase measure.
#
# ## Record and fit the judgments
#
# Suppose the expert gives prior means of 0.2 for the intercept and -0.1 for the slope,
# and a prior median of `exp(-0.3)` (approximately 0.741) for the residual standard deviation.
# These approximate targets use discrepancy scale 1. The vector mean belief contributes
# the average squared error across its two entries.

beliefs = (
    :beta => (Approximate(:mean, Mean(), [0.2, -0.1]; scale=1.0),),
    :sigma => (Approximate(:median, Quantile(0.5), exp(-0.3); scale=1.0),),
)
problem = Elicitation(beliefs, [candidate])
report = fit(Xoshiro(1001), problem, NelderMead(); maxiters=3000, g_tol=1e-14)
validation = validate(Xoshiro(1002), report)
display(validation.checks)

# For this family, the targets correspond to `p.beta = [0.2, -0.1]` and
# `p.log_sigma = -0.3`. Inspect the achieved values in `validation.checks` before accepting
# the fit. Approximate judgments receive status `:target`.

selection = select(
    report,
    (:joint, 1);
    validation,
    rationale="The fitted means and scale median match the elicited targets under the agreed covariance and independence assumptions.",
)
prior = measure(selection)
display(rand(Xoshiro(1004), prior))

# A prior draw is a named tuple with fields `beta` and `sigma`. The selected measure
# retains the coefficient covariance and the positive support of `sigma`.
#
# ## Define the observation model
#
# The conditional mean at `x = 2` is `beta[1] + 2 * beta[2]`. Wren draws the complete
# parameter tuple at `theta ~ prior`, then evaluates the Normal likelihood for `y`.

@wren function regression(prior, y)
    theta ~ prior
    @observe y ~ Normal(theta.beta[1] + 2 * theta.beta[2], theta.sigma)
end

model = regression(prior, 0.4)

# Plaice supplies the coordinate transforms used during inference. The two coefficient
# coordinates retain their values, and the positive scale uses `log(sigma)`.
# Wren evaluates the joint prior in these coordinates, including the transformation Jacobian.
#
# ## Draw posterior samples
#
# NUTS uses Enzyme to differentiate the posterior log density. The pinned Enzyme version
# requires a workaround for copying this prior measure, tracked in
# [Enzyme #3527](https://github.com/EnzymeAD/Enzyme.jl/issues/3527) and
# [#3561](https://github.com/EnzymeAD/Enzyme.jl/issues/3561).
# Since this model treats `prior` as read-only, we disable its defensive copy through
# Wren's internal copy-selection API. Other arguments retain their copy settings.

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

# The result is a FlexiChains chain with 200 retained draws in one chain.
# Each draw contains the joint parameter value `theta`. These draws describe the posterior
# conditional on `y = 0.4`. Assess convergence and Monte Carlo error before reporting estimates.
