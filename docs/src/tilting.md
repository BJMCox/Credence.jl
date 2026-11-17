# [Updating a reference prior](@id tilting-guide)

Bounded-KL tilting adjusts a fixed reference prior to satisfy further probability or
expectation requirements. It minimizes `D(Q || P)` subject to those requirements and
explicit bounds on the density ratio `dQ/dP`. The reference remains fixed throughout
the fit.

## Probability requirements

Consider a standard Normal reference for a standardized calibration offset. The expert
requires a probability between 0.68 and 0.72 that the offset is positive:

```@example tilting
using Credence, MeasureBase, Random

reference = StdNormal()
beliefs = (Hard(:positive, Probability(0, Inf), Between(0.68, 0.72)),)
method = BoundedKL(lower=0.5, upper=2.0)
report = tilt(reference, beliefs, method)
checks = validate(Xoshiro(701), report)
selection =
    select(report; validation=checks, rationale="Positive-offset probability reviewed")
prior = measure(selection)
(
    probability=evaluate(Probability(0, Inf), prior).value,
    logdensity=logdensityof(prior, 1.0),
)
```

The optimum assigns probability 0.68 to the positive half-line. Its density ratio is
constant within the event and its complement. Their reference probabilities determine
the two constants analytically. `rand(Xoshiro(702), prior^4096)` draws independently
from the updated law through bounded rejection.

The lower ratio bound preserves the reference's positive-mass regions. A zero lower
bound permits boundary solutions, including conditioning on an event. Both the event
and its complement constrain the feasible probability range.

## Moment requirements

General requirements use an Optimization.jl algorithm. The dual objective supplies
analytic gradients. A Uniform reference below explicitly assumes equal density on
the unit interval:

```@example tilting
using Optimization, OptimizationOptimJL

moment_report = tilt(
    StdUniform(),
    (Hard(:mean, Mean(), Between(0.55, 0.6)),),
    method,
    Fminbox(BFGS());
    g_tol=1e-10,
)
(
    mean=evaluate(Mean(), measure(moment_report)).value,
    feasibility=moment_report.feasibility,
    optimality=moment_report.optimality,
)
```

`Mean()` and `Probability(...)` describe linear expectations. `Expectation(f)` applies
a pointwise function before averaging. Its absolute reference expectation must exist.
`Statistic(f)` instead receives a measure and retains its separate family-fitting meaning.
Scalar continuous integration requires an atom-free reference with an SMF and inverse.
Known interval boundaries split the integration domain. `Quadrature` sets the error
target and evaluation budget over the full reference probability range.

For an actual finite law, declare its support and relative masses with
`FiniteReference((0.0, 1.0, 2.0), (1, 2, 1))`. Use `IPNewton()` for its constrained
primal problem. The finite solver supplies objective and constraint derivatives and
checks the original requirements after solving.

Finite constraints can force atoms to zero mass or to a ratio cap. The solver removes
coordinates proved fixed by an extremal requirement before optimizing the remaining
masses. Other boundary faces can leave optimality unresolved. The report distinguishes
these cases from a proved infeasible requirement.

## Using the updated prior

The selected measure supports ordinary MeasureBase density evaluation and sampling.
Its Plaice chart uses the reference support. Conditioning on a connected interval
restricts that chart to the retained interval. A disconnected support requires a
separate chart; finite laws have no continuous chart.

The [Wren integration example](@ref wren-tutorial) accepts such a measure as its prior.
For example, the selected Normal event tilt above can be used directly:

```julia
using Wren: @wren, VecModel, FromPrior
using ADTypes: AutoForwardDiff
using LogDensityProblems: logdensity_and_gradient

@wren function calibration_offset(prior)
    offset ~ prior
end

model = VecModel(
    calibration_offset(prior),
    AutoForwardDiff();
    latent_source=FromPrior(Xoshiro(703)),
)
logdensity_and_gradient(model, [0.4])
```

## Saved sessions

Each selection retains its `TiltReport`, including the reference, normalization,
KL divergence, ratio bounds, and projection status. Save the selection with its
validation and rationale, then reload the fitted state:

```@example tilting
restored = mktempdir() do directory
    path = joinpath(directory, "calibration-prior.toml")
    save_session(path, selection; metadata=(parameter=:offset,))
    load_session(path)
end
restored_selection = only(restored.selections)
(
    role=restored_selection.role,
    normalization=restored.report.normalization,
    logdensity=logdensityof(measure(restored_selection), 1.0),
)
```

Loading restores the fitted parameters and retained checks. Numerical normalization
estimates keep their bounds and evidence. Custom expectation functions use the same
versioned callback registry for saving and loading, as in other saved sessions.

Pass `measure(restored_selection)` as the prior in the Wren model above. FlatPPL
export preserves the restored event weights and known normalizer:

```@example tilting
flatppl(restored_selection)
```

Finite laws export as weighted atoms. Affine tilts of `StdUniform()` export as clipped
exponential weights. Arbitrary Julia callbacks require a FlatPPL translator.

## Numerical evidence

The report separates solver results, feasibility, optimality, normalization, and density
ratio bounds. `:within_tolerance` identifies a numerical constraint residual.
`validate(rng, report)` supplies fresh belief checks. Acceptance requires resolved
projection evidence and the ordinary validation checks.

Analytic event updates and affine tilts of `StdUniform()` have known normalizers.
General scalar tilts retain a quadrature estimate separately from their native measure.
For these fits, `logdensity_estimate(report, x)` returns an `Estimate` with estimated
normalization error. Ordinary normalized `logdensityof` remains unavailable while the
normalizer is unknown. Independent rejection draws still target the normalized law.

If the fitted raw weight is `w` and its mass is `Z`, the sampled density ratio is
`w/Z`. The reported ratio bounds propagate the normalization error. Estimated evidence
therefore remains separate from a claim that the original ratio bounds hold exactly.
A nonconverged solver leaves optimality unresolved. An exhausted quadrature budget
raises an error so the integration settings can be revised.

A numerical fit can be retained with `select(report; role=:reference, validation, rationale="...")`. For posterior inference with this fitted prior held fixed,
`MeasureBase.basemeasure(measure(report))` supplies the native unnormalized kernel.
Its missing normalizer is constant across parameter points. Normalized density
evaluation and marginal likelihoods require the normalization evidence separately.
