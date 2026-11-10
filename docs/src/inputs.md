# [Named beliefs and candidate families](@id input-guide)

Each parameter name is paired with a collection of beliefs. Each belief specifies a statistic:

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

`:S` abbreviates `@vn(S)`. Indexed and nested references such as `@vn(B[1])` and
`@vn(detector.width)` use VarNames. A whole vector or matrix can remain one target.
Belief IDs such as `:median` are local labels. The statistic constructor defines their meaning.

The two mean errors above use different scales. An error of one in the first entry
counts equally to an error of two in the second. A scalar scale applies uniformly.
Array targets and array scales must have matching axes. Each array belief contributes
the average squared scaled error across supplied free entries. Importance defaults to one.

Treat input arrays and retained measures as read-only. Use `replace` to revise a belief
and fit a new report. Each report requires its own validation.

Both `nothing` and `missing` leave an entry unspecified. A missing lower bound leaves
the upper bound active, and conversely. A supplied approximate target needs a positive
finite scale. Hard array bounds apply elementwise.

## Declare candidate families separately

```@example inputs
candidates = (
    :S => (
        :beta => ScaledBetaFamily(0.0, 8.0),
        :lognormal => TruncatedFamily(Distributions.LogNormal, 0.0, 8.0),
    ),
    :B => (
        :joint => MvLogNormalFamily(2),
    ),
)
problem = Elicitation(beliefs, candidates)
problem[:S, :median].target
```

Standard family adapters provide valid parameter transforms and recorded starts.
Override `starts` when external knowledge suggests better initial parameters.
`NormalFamily(location=125.0)` fixes the location while fitting its scale.

An array candidate must declare its structure. `IIDFamily(LogNormalFamily(), 2)` shares
one fitted parameter set and declares independent entries. Separate families under
`@vn(B[1])` and `@vn(B[2])` fit separate parameter sets. A multivariate family defines
the vector's joint law.

Custom families retain an explicit builder and named starts:

```@example inputs
custom = Candidate(
    :beta,
    p -> asmeasure(Distributions.LocationScale(
        0.0, 8.0,
        Distributions.Beta(exp(p.log_alpha), exp(p.log_beta)),
    )),
    ((log_alpha=0.5, log_beta=1.0),),
)
nothing # hide
```

Credence converts the named record to optimizer coordinates and restores it for the
builder and fitted result. Existing positional builders remain supported.

## Define joint and derived beliefs

```@example inputs
joint_beliefs = (
    Joint(@vn(B[1]), @vn(B[2])) => (
        Approximate(:covariance, CovarianceMatrix(),
            UpperTriangular([4.0 1.0; 0.0 9.0]);
            scale=UpperTriangular([0.5 0.2; 0.0 1.0])),
    ),
    Derived(:background_sum, :B, B -> sum(B)) => (
        Approximate(:mean, Mean(), 12.0; scale=2.0),
    ),
)
nothing # hide
```

`Joint` reference order defines the matrix coordinates. `Derived` calls its function
with only its declared parameter values, positionally in that order. Derived names must
be unique within a problem.

`Covariance()` and `Correlation()` describe one pair of scalar quantities.
`CovarianceMatrix()` and `CorrelationMatrix()` return full matrices. Covariance matrices
count one triangle including the diagonal. Correlation matrices exclude their fixed unit diagonal.

Triangular wrappers declare the supplied triangle. Full matrices must agree across their
mirrored targets, bounds, scales, and omission patterns. Ordinary matrix-valued statistics
retain ordinary matrix semantics. `diagnostics(problem)` flags inconsistent complete
matrix targets. The original targets remain available for discussion.

## Choose dependence and combinations

Specify dependence when combining marginal families. `Independent` constructs their product:

```@example inputs
joint_problem = Elicitation((beliefs..., joint_beliefs...), candidates;
    combinations=(
        :baseline => Independent(:S => :beta, :B => :joint),
        :alternative => Independent(:S => :lognormal, :B => :joint),
    ),
)
nothing # hide
```

`AllCombinations()` explicitly requests the Cartesian product of marginal family choices.
Combined starts cycle through factor starts instead of multiplying their counts.

Fit and select marginals first when useful. An accepted selection passed as an independent
factor keeps its exact fitted law. `unlock(selection)` releases its fitted parameters while
retaining its hard beliefs. Joint checks remain pending until a candidate covers their references.

Use `sampling=MonteCarlo(n)` on `fit` or `validate` when analytic evaluation is unavailable.
Sampled beliefs share joint draws, and derived quantities reuse their evaluated values.
Fitting replays fixed random inputs. Validation uses a separate fresh batch and records its seed.

The [programmatic workflow](@ref scalar-tutorial) and [guided session](@ref guided-tutorial)
apply this input form to a complete scientific example.
