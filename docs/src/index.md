# Credence.jl

Credence fits candidate prior distributions to an expert's judgments about probabilities,
quantiles, means, and dispersion. It provides methods for comparing families, validating
fitted candidates, and recording a selection. The selected prior is a native MeasureBase
measure, suitable for use in a Wren model.

This initial prototype supports elicitation from a single expert. Family selection is
an explicit expert decision, informed by the fitted targets and candidate comparisons.

## Installation

Credence is not registered. From a source checkout, install its pinned development dependencies:

```sh
git clone https://github.com/BJMCox/Credence.jl.git
cd Credence.jl
julia --project=. dev/setup.jl
```

The core supports Julia 1.10 and later. Use Julia 1.13 or later for the example environments,
which use Pkg's source pins for the development MeasureBase and Wren APIs.

## Check a belief

Suppose `x` has a prior median between -0.1 and 0.1. Compare Normal and LogNormal
families using their declared starts:

```@example start
using Credence, MeasureBase, Random
import Distributions

beliefs = (:x => (Hard(:median, Quantile(0.5), Between(-0.1, 0.1)),),)
candidates = (:x => (
    :normal => NormalFamily(location=0.0),
    :lognormal => LogNormalFamily(location=0.0),
),)
problem = Elicitation(beliefs, candidates)
report = fit(Xoshiro(1), problem)
validation = validate(Xoshiro(2), report)

map(validation.checks) do checks
    check = only(checks)
    (; median=check.estimate.value, status=check.status)
end
```

`fit(rng, problem)` evaluates the supplied starts. Here, the Normal median is 0 and
satisfies the requirement. The LogNormal median is 1 and violates it.
Loading Optimization.jl adds `fit(rng, problem, algorithm; ...)` to optimize family parameters.

## Select and use a prior

Selection records your rationale and retains the validation that supports acceptance:

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

`measure(selection)` returns the selected measure itself. The report stays separate from
the prior. Treat report measures and captured callbacks as read-only after validation.

## Choose a workflow

The [programmatic tutorial](@ref scalar-tutorial) develops a complete elicitation example
for a rare-event search. It defines the model and candidate families, records judgments,
compares fits, revises a judgment, and selects a prior. Six hyperparameters describe signal
strength, location, resolution, and background uncertainty. Five conditional background
rates complete the hierarchy. The tutorial includes all model and plotting definitions.

The [Wren tutorial](@ref wren-tutorial) fits a joint coefficient and scale prior, preserves
coefficient dependence through Plaice, then samples a posterior with NUTS.

[Beliefs and validation](@ref validation-guide) explains hard requirements, approximate
targets, numerical evidence, and acceptance. The [API reference](@ref api-reference)
lists the available constructors and operations.

Predictive fitting, formula macros, tilting, judgment-uncertainty propagation,
portable persistence, interactive controls, and FlatPPL export remain planned.
