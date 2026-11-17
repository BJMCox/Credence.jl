# [Updating a reference prior](@id tilting-guide)

Bounded-KL tilting adjusts a fixed reference prior to satisfy further probability
requirements. It minimizes `D(Q || P)` subject to those requirements and
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

## Using the updated prior

The selected measure supports ordinary MeasureBase density evaluation and sampling.
Its Plaice chart uses the reference support. Conditioning on a connected interval
restricts that chart to the retained interval. A disconnected support requires a
separate chart.

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

FlatPPL export preserves the event weights and the known normalizer:

```@example tilting
flatppl(prior)
```

## Projection evidence

The report separates solver results, feasibility, optimality, normalization, and density
ratio bounds. `:within_tolerance` identifies a numerical constraint residual.
`validate(rng, report)` supplies fresh belief checks. Acceptance requires resolved
projection evidence and the ordinary validation checks.
