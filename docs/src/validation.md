# [Beliefs and validation](@id validation-guide)

## State what the expert means

| Belief                                                  | Meaning                                                                    |
|:------------------------------------------------------- |:-------------------------------------------------------------------------- |
| `Hard(id, statistic, AtLeast(x))`                       | The statistic must be at least `x`.                                        |
| `Hard(id, statistic, AtMost(x))`                        | The statistic must be at most `x`.                                         |
| `Hard(id, statistic, Between(a, b))`                    | The statistic must lie in the closed interval `[a, b]`.                    |
| `Hard(id, statistic, EqualTo(x))`                       | The statistic must equal `x` exactly.                                      |
| `Approximate(id, statistic, target; scale, importance)` | A point target with an explicit discrepancy scale and optional importance. |

Use `Between` to express an expert's tolerance. `EqualTo` supplies no implicit numerical
tolerance. An approximate target records a discrepancy, not a pass/fail requirement.

`Probability(a, b)` means `P(a < X <= b)`, including for discrete laws.
`Quantile(p)` is the lower quantile, so it preserves atoms and support gaps.
`Mean()` and `Std()` require those moments to exist. `Statistic(f; label)` supplies an
analytic statistic of a measure, including properties of a joint prior.

The [input guide](@ref input-guide) covers named parameter groups, joint and derived
quantities, array targets, and standard family descriptors.

## Fit without hiding preferences

Each `Candidate` supplies a measure builder, parameter starts, and optional parameter bounds.
Builders must return normalized MeasureBase measures. A failed start remains in the report
with its error. Successful numerical fits retain the solver result and its return code.

Scalar optimization minimizes the sum of squared, scale-normalized target discrepancies,
weighted by importance. Weights are divided by their maximum first. Multiplying all weights
by a positive constant therefore leaves the objective unchanged. Zero-weight targets are
not fitted. Importance defaults to one per belief. Array beliefs use the average squared
scaled error over their supplied free entries. Splitting one array belief into several
scalar beliefs changes the weighting unless importance is redistributed explicitly.

Hard beliefs become constraints. Choose a solver that supports them, such as `IPNewton()`
from OptimizationOptimJL. The default AD choice is `Optimization.NoAD()`; pass a supported
AD type when the solver needs derivatives. The tutorials use deterministic solvers.
Stochastic backends need their own explicitly configured RNG through their solver options.

`compare(report)` uses each approximate target's absolute, scale-normalized discrepancy.
It retains Pareto dominance, ties, tradeoffs, and unresolved pairs. Importance weights do not
collapse these comparisons into one score. Satisfying a hard requirement by a larger margin
does not make a candidate preferable.

Ungrouped and combined results use `(candidate_id, start_index)`. Marginal results use
`(target, candidate_id, start_index)`. `select(report, target, (candidate_id, start_index); ...)`
provides a convenient marginal lookup.

| Field          | Contents                                                          |
|:-------------- |:----------------------------------------------------------------- |
| `feasible`     | IDs with all hard requirements satisfied and all targets defined. |
| `dominates`    | Ordered `(better, worse)` result pairs by target discrepancy.     |
| `ties`         | Pairs with equal discrepancy vectors.                             |
| `incomparable` | Pairs where improving one target worsens another.                 |
| `unresolved`   | Pairs whose evidence cannot support an exact comparison.          |

`Evaluate()` considers only supplied points. Optimization retains the solution from each
start. Neither operation claims a complete feasible region or a complete Pareto frontier.

## Check fixed candidates

`validate(rng, report)` evaluates the retained measures afresh. Analytic measure operations
provide the default evidence. Failed or unavailable operations produce undefined checks.

For event probabilities, `MonteCarlo(n; alpha)` uses independent draws and Hoeffding bounds:

```@example validation
using Credence, MeasureBase, Random

belief = Hard(:lower_half, Probability(-Inf, 0), Between(0.4, 0.6))
candidate = Candidate(:normal, _ -> StdNormal(), [0.0])
report = fit(Xoshiro(10), Elicitation((belief,), (candidate,)))
validation = validate(Xoshiro(11), report; plan=MonteCarlo(10_000; alpha=0.01))
check = only(only(validation.checks))
(; estimate=check.estimate, status=check.status)
```

The error budget is divided across all hard checks in this report. The bounds require a valid
independent sampler. The `sampling=MonteCarlo(n)` option keeps analytic evaluators where
available and samples remaining quantities using one shared joint batch per candidate.
Event probabilities use Hoeffding bounds, and quantiles use DKW enclosures. Sampled means
and dispersion statistics report estimated intervals, which cannot certify hard constraints.
`estimate.n` records the draw count. The report and validation retain their sampling seeds.
Repeated adaptive validation needs a separate error-budget policy. The caller owns the seeds.

An enclosure wholly inside a hard requirement satisfies it. A disjoint enclosure violates it.
Overlap remains inconclusive. Evidence marked only `:estimated` cannot certify satisfaction
or violation. Uncertain approximate-target comparisons remain unresolved.

## Accept explicitly

`select(report, id; validation, rationale)` accepts a concrete result only when the validation
belongs to that same report and every hard requirement is satisfied. Approximate targets stay
targets, so review their achieved values yourself. Solver success does not replace validation.

Use `role=:reference` to retain a result for exploration without declaring acceptance.
Selection records the role and rationale. `measure(selection)` returns the native measure
without adding report metadata to the prior.

Marginal acceptance is local. Joint beliefs stay `:pending` until one candidate covers their
references. Acceptance of a complete joint prior requires those beliefs to be checked too.
An accepted selection passed to `Independent` retains that exact marginal law. Calling
`unlock(selection)` restores its fitted family start and retains its hard beliefs.
