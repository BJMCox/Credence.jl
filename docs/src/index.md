# Credence.jl

```@docs
Credence
```

## Prototype workflow

Define beliefs with `Hard` requirements or `Approximate` targets. Supply `Candidate` builders
that return normalized MeasureBase measures, then construct an `Elicitation` problem.

`fit(rng, problem)` evaluates the declared parameter points. Loading Optimization.jl adds
`fit(rng, problem, algorithm; ...)` for numerical fitting. Multiple approximate targets need
explicit importance weights for scalar optimization. Hard beliefs become solver constraints.
Use a constrained optimizer when requirements are present.

```@example core
using Credence, MeasureBase, Random

belief = Hard(:median, Quantile(0.5), Between(-0.1, 0.1))
candidates = (
    Candidate(:normal, _ -> StdNormal(), [0.0]),
    Candidate(:exponential, _ -> StdExponential(), [0.0]),
)
report = fit(Xoshiro(1), Elicitation((belief,), candidates))
validation = validate(Xoshiro(2), report)
choice =
    select(report, (:normal, 1); validation, rationale="Median lies in the stated range")
rand(Xoshiro(3), measure(choice)^5)
```

`compare(report)` retains dominance, ties, tradeoffs, and unresolved comparisons.
`validate(rng, report)` checks the fixed results. `select` requires explicit validation and
a rationale before accepting a prior. `measure(selection)` returns the native measure.

`Probability(a,b)` means `P(a < X <= b)`, including for discrete laws.
`Quantile(p)` uses the lower quantile, preserving jumps and support gaps.
`Statistic(f; label)` supplies an analytic oracle for other properties of a joint prior.

## Evidence and limits

`Analytic()` calls measure operations. `MonteCarlo(n; alpha)` validates event probabilities
with Hoeffding bounds from independent draws, dividing the error budget across this report's
hard checks. It requires a valid independent sampler and an explicit RNG.
Repeated adaptive validation needs a separate error budget. Seeds remain caller-owned.
Estimated diagnostics cannot certify hard constraints, and uncertain comparisons stay unresolved.

Solver return codes are kept separately from validation. A failed start remains in the report.
`Evaluate()` enumerates only supplied points. Numerical fitting retains the supplied starts'
solutions, without claiming a complete feasible region or Pareto frontier.
Treat report measures and captured callbacks as read-only after validation.

The Optimization extension defaults to deterministic solvers. A stochastic backend needs
its own explicitly configured RNG, passed through the solver's options.

## Wren and Plaice

Loading Plaice activates vectorization and support charts for MeasureBase's wrapped laws
and named products. Use one joint prior site in Wren to retain coefficient dependence.
The integration checks vector round trips, Jacobians, and gradients with ForwardDiff and Enzyme,
then runs Enzyme NUTS. It does not claim general smooth charts for arbitrary weighted or mixed measures.

The Enzyme example avoids [a deepcopy-rule failure](https://github.com/EnzymeAD/Enzyme.jl/issues/3527)
by removing only its fixed, read-only prior from Wren's copy set. The related
[mixed-return adjoint issue](https://github.com/EnzymeAD/Enzyme.jl/issues/3561) also needs attention upstream.
The workaround uses the pinned Wren version's internal API. It does not disable copies of mutable arguments:
Wren can write latent values into those buffers, and removing their copies can lose gradients.

## Planned work

Predictive fitting, formula macros, tilting, marginal locks, uncertainty propagation, persistence,
notebook interaction, and FlatPPL export belong to the full design and remain unimplemented.

## API

```@autodocs
Modules = [Credence]
Private = false
```
