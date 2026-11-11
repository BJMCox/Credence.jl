# # [Using an elicited prior in Wren](@id wren-tutorial)
#
# This example uses an elicited prior in a Normal regression. The parameters are the
# intercept `beta[1]`, slope `beta[2]`, and residual standard deviation `sigma`.
# We fit their prior to expert judgments, then observe `y = 0.4` at `x = 2`.
#
# Run the example from the package directory with Julia 1.13 or later:
#
# ```sh
# julia --project=test/integration -e 'using Pkg; Pkg.instantiate()'
# julia --project=test/integration examples/wren.jl
# ```
#
# The environment pins development versions of Wren, Plaice, and MeasureBase.
# Installing Wren requires access to its private repository.

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

# ## Define the prior
#
# The coefficients have a bivariate Normal prior with fitted mean `p.beta` and fixed
# covariance `covariance`. Independently, `sigma` has a LogNormal prior with fitted
# log location `p.log_sigma` and fixed log standard deviation 0.4.
# Thus fitting changes three location parameters. The covariance and spread are family assumptions.

covariance = [1.0 0.4; 0.4 0.8]
function build_prior(p)
    MeasureBase.productmeasure((
        beta=MeasureBase.asmeasure(MvNormal(p.beta, covariance)),
        sigma=MeasureBase.asmeasure(LogNormal(p.log_sigma, 0.4)),
    ))
end
candidate = Candidate(:joint, build_prior, ((beta=zeros(2), log_sigma=0.0),))

# The builder returns a named product measure with fields `beta` and `sigma`.
# `Candidate` provides its identifier and starting parameter record.
#
# ## Fit the judgments
#
# Suppose the expert gives mean 0.2 for the intercept, mean -0.1 for the slope, and
# median `exp(-0.3)`, approximately 0.741, for `sigma`. The targets below use scale 1.
# The vector mean target contributes the average squared error across its two entries.

beliefs = (
    :beta => (Approximate(:mean, Mean(), [0.2, -0.1]; scale=1.0),),
    :sigma => (Approximate(:median, Quantile(0.5), exp(-0.3); scale=1.0),),
)
problem = Elicitation(beliefs, [candidate])
report = fit(Xoshiro(1001), problem, NelderMead(); maxiters=3000, g_tol=1e-14)
validation = validate(Xoshiro(1002), report)
display(belief_summary(validation))

# These targets correspond to `p.beta = [0.2, -0.1]` and `p.log_sigma = -0.3`.
# Check the achieved statistics in `belief_summary(validation)`. Their status is `:target` because
# the beliefs specify approximate targets. Selection records acceptance of this fit and its assumptions.

selection = select(
    report,
    (:joint, 1);
    validation,
    rationale="The fitted means and scale median match the elicited targets under the agreed covariance and independence assumptions.",
)
prior = measure(selection)
display(rand(Xoshiro(1004), prior))

# A draw from `prior` is a named tuple containing `beta` and `sigma`.
#
# ## Condition on an observation
#
# At `x = 2`, the conditional mean is `beta[1] + 2 * beta[2]`.
# The statement `theta ~ prior` draws the joint parameter tuple, preserving coefficient
# dependence. The observation statement supplies the Normal likelihood for `y`.

@wren function regression(prior, y)
    theta ~ prior
    @observe y ~ Normal(theta.beta[1] + 2 * theta.beta[2], theta.sigma)
end

model = regression(prior, 0.4)

# Plaice maps the prior to the coordinates used for inference: the two coefficient values
# and `log(sigma)`. Wren includes the transformation Jacobian when evaluating the density.
#
# ## Sample the posterior
#
# NUTS uses Enzyme to differentiate the posterior log density. The pinned Enzyme version
# has a copy issue for this measure, tracked in
# [Enzyme #3527](https://github.com/EnzymeAD/Enzyme.jl/issues/3527) and
# [#3561](https://github.com/EnzymeAD/Enzyme.jl/issues/3561).
# This model reads `prior` without mutation, so its defensive copy can be disabled through
# Wren's internal API. The remaining arguments keep their copy settings.

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

# `chain` contains 200 posterior draws in one FlexiChains chain, conditional on `y = 0.4`.
# Each draw contains `theta`, with its coefficient vector and residual standard deviation.
# Assess convergence and Monte Carlo error before reporting posterior estimates.
