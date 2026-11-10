# [Named beliefs and candidate families](@id input-guide)

An elicitation problem pairs parameter references with beliefs and candidate families.
For example, `S` below is a scalar and `B` is a vector with two entries:

```@example inputs
using Credence, MeasureBase, VarNames, LinearAlgebra
import Distributions

beliefs = (
    :S => (
        Approximate(:median, Quantile(0.5), 1.2; scale=1.0),
        Approximate(:upper, Quantile(0.95), 3.5; scale=1.0),
    ),
    :B => (
        Approximate(:mean, Mean(), [5.0, 7.0]; scale=[1.0, 2.0]),
        Hard(:mean_bounds, Mean(), Between([nothing, 4.0], [8.0, missing])),
    ),
)
nothing # hide
```

A symbol such as `:S` is shorthand for `@vn(S)`. Use VarNames for indexed or nested
references, such as `@vn(B[1])` or `@vn(detector.width)`. Each belief has a local identifier,
a statistic, and a target or bound. Thus `:median` identifies the belief, while
`Quantile(0.5)` defines the statistic being elicited.

## Array targets and scales

The mean target for `B` is `[5, 7]`, with discrepancy scales `[1, 2]`. An error of one
in the first mean therefore has the same scaled size as an error of two in the second.
A scalar scale applies to every entry. Array targets and scales must have matching axes.
Scales for supplied approximate targets must be positive and finite.

Use `nothing` or `missing` for an unspecified entry. Bounds act elementwise: an omitted
lower bound still permits an upper bound on that entry. Each array belief contributes
the average squared scaled error over its supplied free entries. Its default importance is one.

Treat input arrays as read-only. To revise a judgment, use `replace`, fit the revised
problem, and validate the resulting report.

## Candidate families

Use the same parameter references to associate candidate families with the beliefs:

```@example inputs
candidates = (
    :S => (
        :beta => ScaledBetaFamily(0.0, 8.0),
        :lognormal => TruncatedFamily(Distributions.LogNormal, 0.0, 8.0),
    ),
    :B => (:joint => MvLogNormalFamily(2),),
)
problem = Elicitation(beliefs, candidates)
problem[:S, :median].target
```

Family descriptors supply starting parameters and the transforms needed to keep parameters
in their domains. Set `starts` to supply other starting values. Fixed keywords restrict the
family: `NormalFamily(location=125.0)` fits the scale of a Normal with mean 125.

For arrays, `IIDFamily(LogNormalFamily(), 2)` fits one law shared by two independent entries.
Separate candidates for `@vn(B[1])` and `@vn(B[2])` fit their parameters separately.
`MvLogNormalFamily(2)` instead fits a joint law for the two-dimensional vector.

A custom `Candidate` supplies a measure builder and starting parameter records:

```@example inputs
custom = Candidate(
    :beta,
    p -> asmeasure(
        Distributions.LocationScale(
            0.0,
            8.0,
            Distributions.Beta(exp(p.log_alpha), exp(p.log_beta)),
        ),
    ),
    ((log_alpha=0.5, log_beta=1.0),),
)
nothing # hide
```

Here the parameters are log shapes, so exponentiation gives positive Beta shapes. Credence
flattens the record for optimization and restores its names when calling the builder and
recording the fitted parameters. Builders can also accept positional parameter vectors.

## Joint and derived quantities

Place beliefs about several quantities under `Joint`. Use `Derived` for a function of
parameter values:

```@example inputs
joint_beliefs = (
    Joint(@vn(B[1]), @vn(B[2])) => (
        Approximate(
            :covariance,
            CovarianceMatrix(),
            UpperTriangular([4.0 1.0; 0.0 9.0]);
            scale=UpperTriangular([0.5 0.2; 0.0 1.0]),
        ),
    ),
    Derived(:background_sum, :B, B -> sum(B)) =>
        (Approximate(:mean, Mean(), 12.0; scale=2.0),),
)
nothing # hide
```

The order of `Joint` references determines the matrix coordinates. `Derived` passes the
selected values to its function in reference order. Derived names must be unique within
the problem.

`Covariance()` and `Correlation()` describe two scalar quantities. Their matrix forms,
`CovarianceMatrix()` and `CorrelationMatrix()`, describe a vector or a `Joint` selection.
Covariance targets count one triangle, including the diagonal. Correlation targets count
one triangle with the fixed unit diagonal excluded.

`UpperTriangular` and `LowerTriangular` identify the supplied triangle. In a full symmetric
target, mirrored entries must agree in value, bounds, scales, and omission patterns.
Other matrix-valued statistics use every supplied entry. `diagnostics(problem)` reports
inconsistent complete covariance or correlation targets and preserves the submitted values.

## Combining families

Specify which marginal families form each joint candidate. `Independent` constructs a
product measure:

```@example inputs
joint_problem = Elicitation(
    (beliefs..., joint_beliefs...),
    candidates;
    combinations=(
        :baseline => Independent(:S => :beta, :B => :joint),
        :alternative => Independent(:S => :lognormal, :B => :joint),
    ),
)
nothing # hide
```

The first combination uses the Beta candidate for `S` and the multivariate LogNormal
candidate for `B`. Independence applies between `S` and the vector `B`. The multivariate
family determines dependence within `B`.

`AllCombinations()` constructs the Cartesian product of marginal family choices.
For each combination, starting values cycle through the factor starts rather than taking
their Cartesian product.

An accepted marginal selection can serve as a fixed factor in `Independent`. Call
`unlock(selection)` to fit its family parameters again while retaining its hard beliefs.
Joint checks remain `:pending` until a candidate includes all their references.

## Sampled statistics

Pass `sampling=MonteCarlo(n)` to `fit` or `validate` to sample statistics that lack analytic
evaluators. Beliefs for one candidate share a batch of joint draws. Derived quantities
reuse the values computed from those draws.

Fitting reuses fixed random inputs to evaluate its objective. Validation draws a fresh
batch and records its seed. See [Beliefs and validation](@ref validation-guide) for the
uncertainty attached to sampled statistics.
