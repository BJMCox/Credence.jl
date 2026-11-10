# [Beliefs and validation](@id validation-guide)

A belief states either a requirement on a statistic or an approximate target for it.
Requirements determine whether a candidate is acceptable. Approximate targets measure
how closely a family represents the elicited judgments.

## Statistics and requirements

| Belief | Interpretation |
|:--|:--|
| `Hard(id, statistic, AtLeast(x))` | The statistic is at least `x`. |
| `Hard(id, statistic, AtMost(x))` | The statistic is at most `x`. |
| `Hard(id, statistic, Between(a, b))` | The statistic lies in the closed interval `[a, b]`. |
| `Hard(id, statistic, EqualTo(x))` | The statistic equals `x` exactly. |
| `Approximate(id, statistic, target; scale, importance)` | Fit a point target using the given discrepancy scale and importance. |

Use `Between` for an elicited tolerance and `EqualTo` for exact equality.
`Probability(a, b)` denotes `P(a < X <= b)`, including for discrete measures.
`Quantile(p)` denotes the lower quantile, with atoms and support gaps retained.
`Mean()` and `Std()` apply when their population moments exist.
`Statistic(f; label)` evaluates a custom analytic function of the measure.

[Named inputs](@ref input-guide) explains how to attach these beliefs to scalar, array,
joint, and derived quantities.

## Fitting

Each candidate defines a normalized MeasureBase measure as a function of its parameters.
It also supplies starting values and optional parameter bounds. The report retains every
start, together with its fitted parameters and solver result, or its evaluation error.

For each scalar target, divide the discrepancy by its scale and square the result.
The objective sums these squared discrepancies, weighted by importance. Importance
weights are first divided by their maximum, so a common positive rescaling leaves the
objective unchanged. A zero weight contributes zero, and the default weight is one.
For an array belief, the contribution is the average squared scaled discrepancy over its
supplied free entries. Splitting an array into scalar beliefs therefore requires an
appropriate redistribution of importance to preserve the objective.

Hard beliefs become optimization constraints. Use a solver that supports those constraints,
such as `IPNewton()` from OptimizationOptimJL. Credence defaults to `Optimization.NoAD()`.
Supply an AD type when the chosen solver requires derivatives. Configure the RNG for a
stochastic solver through that solver's options.

`Evaluate()` evaluates the supplied starts. An optimization algorithm instead retains a
solution from each start. Subsequent comparisons concern this finite set of results.

## Comparing candidates

`compare(report)` compares absolute, scale-normalized discrepancies separately for each
approximate target. Importance weights control fitting, while comparison uses the full
discrepancy vector. Hard requirements determine feasibility.

| Field | Contents |
|:--|:--|
| `feasible` | Results satisfying every hard requirement, with every target defined. |
| `dominates` | Ordered pairs in which the first result improves at least one discrepancy and worsens none. |
| `ties` | Pairs with equal discrepancy vectors. |
| `incomparable` | Pairs that trade improvement in one target for deterioration in another. |
| `unresolved` | Pairs whose evidence leaves the comparison uncertain. |

Ungrouped and combined result IDs have the form `(candidate_id, start_index)`.
Marginal result IDs also include the target reference. The form
`select(report, target, (candidate_id, start_index); ...)` selects a marginal result directly.

## Validation

`validate(rng, report)` checks the retained measures again. By default, it uses analytic
measure operations. An unavailable or failed operation gives an undefined check.

For sampled event probabilities, `MonteCarlo(n; alpha)` draws independent samples and
constructs Hoeffding bounds:

```@example validation
using Credence, MeasureBase, Random

belief = Hard(:lower_half, Probability(-Inf, 0), Between(0.4, 0.6))
candidate = Candidate(:normal, _ -> StdNormal(), [0.0])
report = fit(Xoshiro(10), Elicitation((belief,), (candidate,)))
validation = validate(Xoshiro(11), report; plan=MonteCarlo(10_000; alpha=0.01))
check = only(only(validation.checks))
(; estimate=check.estimate, status=check.status)
```

The error budget `alpha` is divided across all hard checks in the report. These bounds
assume independent draws from the candidate measure. Quantile checks use DKW enclosures.
Sampled means and dispersion statistics have estimated intervals, which cannot certify
hard requirements.

With `sampling=MonteCarlo(n)`, analytic evaluators remain in use where available. Other
statistics share one joint sample batch per candidate. `estimate.n` records the sample
count, and reports retain their sampling seeds. Callers control the seeds and the error
budget for repeated adaptive validation.

A check satisfies a hard requirement when its entire enclosure lies within the required
region. A disjoint enclosure violates the requirement. Partial overlap is inconclusive.
Evidence marked `:estimated` supports estimation rather than certification. Approximate
comparisons with uncertain ordering remain unresolved.

## Selection

`select(report, id; validation, rationale)` records acceptance of a fitted result.
The validation must belong to that report and satisfy every hard requirement.
Review approximate targets by their achieved values, even when the solver reports success.
Use `role=:reference` to retain a candidate for further discussion.

`measure(selection)` returns the selected native measure. The selection retains its role,
rationale, and validation separately from the prior.

A marginal selection accepts its own beliefs. Acceptance of a complete joint prior also
requires validation of the joint beliefs. An accepted selection used in `Independent`
keeps its fitted law fixed. `unlock(selection)` permits refitting and retains its hard beliefs.
