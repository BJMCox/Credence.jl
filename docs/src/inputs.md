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
`NormalFamily(scale=1.0)` instead fits its location. Supplying both fixes the law.
`LogNormalFamily` uses the same keywords for log location and log standard deviation.
The [family catalogue](@ref family-catalogue) lists the available families, supports,
parameter conventions, and starting coordinates.

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

A single named start can be passed directly as `(log_alpha=0.5, log_beta=1.0)`.
Use a tuple of such records to supply several starts.

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

Diagnostics also identify reversed quantile judgments and probabilities that contradict
event inclusion. For example, a probability assigned to `(0, 1]` must be no larger than
the probability assigned to `(-1, 2]`. A diagnostic identifies the pair of beliefs involved.
Approximate targets remain available for fitting and discussion after a conflict is reported.

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

## FlatPPL export

`flatppl(selection)` returns a FlatPPL expression for the selected measure.
`flatppl(prior)` accepts a MeasureBase measure directly. Export follows the measure's
structure, so a custom candidate builder can use the same translations as a standard family.

Translations cover standard scalar laws, supported Distributions wrappers, named products,
scalar-factor vectors, and iid arrays. Multivariate Normal and LogNormal translations
preserve covariance. Truncated laws retain their normalization and support bounds.
A measure with an unsupported transform or component requires an additional translation
method. The error identifies that component's path.

The [FlatPPL example](@ref flatppl-tutorial) exports an elicited joint prior and uses it
in a complete observation model, with a density check against Julia.

## Eliciting observations

`Predictive` attaches beliefs to observations generated from uncertain parameters.
Its kernel defines a conditional observation measure. Fitting changes the prior family
parameters while retaining that observation model.

```@example predictive
using Credence, MeasureBase, Optimization, OptimizationOptimJL, Random
using Distributions: Normal

reading = Predictive(:reading, :offset, offset -> asmeasure(Normal(offset, 0.25)))
problem = Elicitation(
    (reading => (Approximate(:mean, Mean(), 2.0; scale=0.2),),),
    (:offset => (:laplace => LaplaceFamily(scale=0.5),),),
)
report = fit(Xoshiro(91), problem, NelderMead(); sampling=MonteCarlo(1024), maxiters=300)
validation = validate(Xoshiro(92), report; sampling=MonteCarlo(4096))
belief_summary(validation)
```

Each independent prior draw produces one observation. Beliefs about the same predictive
target share that observation batch. Fitting uses fixed random inputs to compare parameter
settings. Validation draws a new batch. `MonteCarlo(n)` counts joint parameter/observation draws.

For an existing simulator, use the explicit RNG form:

```@example predictive
simulated_reading = Predictive(
    :simulated_reading,
    :offset;
    simulator=(rng, offset) -> offset + 0.25randn(rng),
)
nothing # hide
```

The simulator can return a scalar or an array. Event predicates can inspect structured
outcomes. Simulation errors retain the predictive target name and the parameter values
that caused the failure. The [Wren example](@ref wren-tutorial) uses a Wren observation
model through this simulator interface.

## Saving and resuming an elicitation

`save_session` writes a versioned TOML record of a problem, fitted parameters, validation,
selections, and revision history. `load_session` restores those objects and rebuilds the
selected measures from their saved parameters.

```@example sessions
using Credence, MeasureBase, Optimization, OptimizationOptimJL, Random
using Distributions: mean

initial = Elicitation(
    (:rate => (Approximate(:mean, Mean(), 6.0; scale=1.0),),),
    (:rate => (:gamma => GammaFamily(shape=2.0),),),
)
initial_report = fit(Xoshiro(101), initial, NelderMead(); maxiters=1000)
initial_validation = validate(Xoshiro(102), initial_report)
revised = replace(initial, :rate => Approximate(:mean, Mean(), 7.5; scale=1.0))
report = fit(Xoshiro(103), revised, NelderMead(); maxiters=1000)
validation = validate(Xoshiro(104), report)
selection = select(
    report,
    :rate,
    (:gamma, 1);
    validation,
    rationale="The revised mean reflects the calibration review.",
)

mkpath("elicitation-output")
path = "elicitation-output/rate-session.toml"
save_session(
    path,
    revised;
    report,
    validation,
    selections=(selection,),
    history=((
        problem=initial,
        report=initial_report,
        validation=initial_validation,
        selections=(),
        rationale="Initial calibration judgment.",
    ),),
    metadata=(rate_unit="events/hour",),
)
restored = load_session(path)
mean(measure(only(restored.selections)))
```

[Download the saved session](elicitation-output/rate-session.toml).
The restored validation refers to the restored report, and the selected prior retains its
rationale. Earlier fitted revisions remain available through `restored.history`.
Optimizer provenance retains the solver type and return code as portable data.

Custom functions use a named callback registry. Supply the same reconstruction recipes
when saving and loading:

```@example sessions
using Distributions: Poisson

simulate_detector(rng, rate) = rand(rng, Poisson(3rate))
counts = Predictive(:counts, :rate; simulator=simulate_detector)
predictive_problem = Elicitation(
    (counts => (Approximate(:mean, Mean(), 6.0; scale=1.0),),),
    (:rate => (:gamma => GammaFamily(shape=2.0, scale=1.0),),),
)
predictive_report = fit(Xoshiro(105), predictive_problem; sampling=MonteCarlo(1024))
predictive_validation = validate(Xoshiro(106), predictive_report)
callbacks = (detector=(callback=simulate_detector, version="1"),)
callback_path = "elicitation-output/predictive-session.toml"
save_session(
    callback_path,
    predictive_problem;
    report=predictive_report,
    validation=predictive_validation,
    callbacks,
)
restored_predictive = load_session(callback_path; callbacks)
belief_summary(restored_predictive.validation)
```

Callback fingerprints cover available source code and serializable captured values.
Explicit versions identify REPL functions and opaque context. Keep those versions aligned
with model dependencies and global data, which source fingerprints alone cannot describe.
Loading checks the session schema, package provenance, and callback fingerprints before
restoring validation. Changed models require a new fit and validation.
