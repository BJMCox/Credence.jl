# Credence.jl

Prior elicitation expresses expert knowledge as a probability distribution. Credence fits
several distribution families to judgments about probabilities, quantiles, means, and
standard deviations. The expert reviews their agreement with those judgments and selects
a prior, recording the reason for the choice.

The package supports a single expert. Priors use MeasureBase and can enter Wren models
through Plaice.

## Installation

Install Credence from the Julia REPL with Julia 1.13.

```julia
using Pkg
Pkg.add(url="https://github.com/BJMCox/Credence.jl.git")
```

## A first example

Add Distributions to use the built-in distribution families.

```julia
Pkg.add("Distributions")
```

Suppose the prior median of `x` must lie between -0.1 and 0.1. Represent this requirement
with `Hard` and compare standard Normal and Exponential priors.

```@example start
using Credence, Distributions, Random

beliefs = (:x => (Hard(:median, Quantile(0.5), Between(-0.1, 0.1)),),)
candidates = (
    :x => (
        :normal => NormalFamily(location=0.0, scale=1.0),
        :exponential => ExponentialFamily(scale=1.0),
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

`fit` evaluates each candidate at its declared parameters. The Normal median
is 0 and satisfies the requirement. The Exponential median is `log(2)` and falls outside
the interval.

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

Each example starts with its own judgments and candidate priors. Run the Julia scripts
from the package directory with Julia 1.13 or later. Wren uses the environment listed
on its example page.

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/pump_repair.jl
```

| Level         | Examples                                                                                                                                        |
|:------------- |:----------------------------------------------------------------------------------------------------------------------------------------------- |
| Easy          | [Check a calibration tolerance](@ref zero-offset-tutorial) and [fit two families to repair times](@ref pump-repair-tutorial).                   |
| Intermediate  | [Revise and save a monitor calibration](@ref monitor-calibration-tutorial) and [fit a vector of source fractions](@ref water-sources-tutorial). |
| Advanced      | [Check prior predictions for a rare-event search](@ref scalar-tutorial) and [sample a posterior in Wren](@ref wren-tutorial).                   |
| Very advanced | [Compare compound shrinkage priors](@ref shrinkage-tutorial) and [export a complete FlatPPL model](@ref flatppl-tutorial).                      |

[Named inputs](@ref input-guide) describes parameter references, arrays, and joint beliefs.
[Beliefs and validation](@ref validation-guide) defines the fitting objective, candidate
comparisons, and the evidence required for acceptance.
[Tilting an existing prior](@ref tilting-guide) covers further probability or expectation
requirements for a fixed reference. The [API reference](@ref api-reference) documents the constructors
and methods.

## Development status

Credence is a prototype. Planned work includes formula syntax,
propagation of judgment uncertainty, and interactive controls.
