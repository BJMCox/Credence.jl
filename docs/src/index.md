# Credence.jl

Prior elicitation expresses expert knowledge as a probability distribution. Credence fits
several distribution families to judgments about probabilities, quantiles, means, and
standard deviations. The expert reviews their agreement with those judgments and selects
a prior, recording the reason for the choice.

The package supports a single expert. Priors use MeasureBase and can enter Wren models
through Plaice.

## Installation

Install the prototype from its source repository:

```sh
git clone https://github.com/BJMCox/Credence.jl.git
cd Credence.jl
julia --project=. dev/setup.jl
```

The package supports Julia 1.10 and later. The examples require Julia 1.13 or later because
their environments pin development versions of MeasureBase and Wren through Pkg source entries.

## A first example

Suppose the prior median of `x` must lie between -0.1 and 0.1. Represent this requirement
with `Hard`, and consider Normal and LogNormal families:

```@example start
using Credence, MeasureBase, Random
import Distributions

beliefs = (:x => (Hard(:median, Quantile(0.5), Between(-0.1, 0.1)),),)
candidates = (
    :x => (
        :normal => NormalFamily(location=0.0),
        :lognormal => LogNormalFamily(location=0.0),
    ),
)
problem = Elicitation(beliefs, candidates)
report = fit(Xoshiro(1), problem)
validation = validate(Xoshiro(2), report)

map(validation.checks) do checks
    check = only(checks)
    (; median=check.estimate.value, status=check.status)
end
```

With no optimization algorithm supplied, `fit` evaluates each family's starting parameters.
The Normal median is 0 and satisfies the requirement. The LogNormal median is 1 and falls
outside the interval.

After reviewing these checks, select the Normal result and record the reason:

```@example start
selection = select(
    report,
    :x,
    (:normal, 1);
    validation,
    rationale="The median satisfies the expert's stated range.",
)
prior = measure(selection)
rand(Xoshiro(3), prior^5)
```

`measure(selection)` returns the prior as a MeasureBase measure. The fitting report and
validation remain available separately. Treat the measures and callbacks in a validated
report as read-only, so that the validation continues to describe the selected prior.

For numerical fitting, load Optimization.jl and call `fit(rng, problem, algorithm; ...)`.

## Examples and guides

The [signal-background example](@ref scalar-tutorial) defines a rare-event model and elicits
priors for its six hyperparameters. It includes candidate families, a revised judgment,
prior selection, and prior predictive checks, with all definitions on the page.

The [Wren example](@ref wren-tutorial) elicits a joint regression prior and uses it to
sample a posterior with NUTS.

[Named inputs](@ref input-guide) describes parameter references, arrays, and joint beliefs.
[Beliefs and validation](@ref validation-guide) defines the fitting objective, candidate
comparisons, and the evidence required for acceptance. The [API reference](@ref api-reference)
contains the constructor and method documentation.

## Development status

Credence is a prototype. Planned work includes predictive fitting, formula syntax, tilting,
propagation of judgment uncertainty, portable persistence, interactive controls, and FlatPPL export.
