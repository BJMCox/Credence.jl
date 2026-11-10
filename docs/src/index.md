# Credence.jl

Credence turns an expert's beliefs into candidate prior distributions. State probability,
quantile, mean, or spread beliefs, compare several families, then select a prior explicitly.
The result is a native MeasureBase measure that can enter a Wren model.

This is an initial prototype for a single expert. Candidate comparison supports expert
judgment. It does not choose a family for you.

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

Suppose the prior's median should lie between -0.1 and 0.1. Compare a standard Normal
and a standard Exponential distribution at their declared parameter points:

```@example start
using Credence, MeasureBase, Random

belief = Hard(:median, Quantile(0.5), Between(-0.1, 0.1))
candidates = (
    Candidate(:normal, _ -> StdNormal(), [0.0]),
    Candidate(:exponential, _ -> StdExponential(), [0.0]),
)
problem = Elicitation((belief,), candidates)
report = fit(Xoshiro(1), problem)
validation = validate(Xoshiro(2), report)

map(validation.checks) do checks
    check = only(checks)
    (; median=check.estimate.value, status=check.status)
end
```

`fit(rng, problem)` evaluates the supplied points. It does not optimize them. The Normal
median satisfies the requirement. The Exponential median does not.

## Select and use a prior

Selection records your rationale and retains the validation that supports acceptance:

```@example start
selection = select(
    report,
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

The [programmatic workflow](@ref scalar-tutorial) runs a recorded set of judgments and an
explicit decision as a script. The [guided REPL session](@ref guided-tutorial) asks one
question at a time, shows feedback, and revises an answer before accepting a prior.
Both use the same constructors, fitting methods, validation, and selection rules.

Loading Optimization.jl adds `fit(rng, problem, algorithm; ...)`. Both workflows compare
Normal and Student-t families rather than choosing a family automatically.

The [Wren tutorial](@ref wren-tutorial) fits a joint coefficient and scale prior, preserves
coefficient dependence through Plaice, then samples a posterior with NUTS.

[Beliefs and validation](@ref validation-guide) explains hard requirements, approximate
targets, numerical evidence, and acceptance. The [API reference](@ref api-reference)
lists the available constructors and operations.

Predictive fitting, formula macros, tilting, marginal locks, judgment-uncertainty propagation,
portable persistence, interactive controls, and FlatPPL export remain planned.
