const NamedProductMeasure = MeasureBase.ProductMeasure{<:NamedTuple}
const JointIntervalProbability = Probability{<:JointIntervalEvent}
const JointIntervalTiltKernel = MeasureBase.DensityMeasure{
    <:MeasureBase.LogFuncDensity{<:EventTiltScore{<:JointIntervalEvent}},
}

function _joint_interval_mass(reference::NamedProductMeasure, event, other=event)
    parts = marginals(reference)
    for name in (keys(event.intervals)..., keys(other.intervals)...)
        haskey(parts, name) ||
            throw(ArgumentError("joint probability names unknown parameter $name"))
    end
    prod(keys(parts)) do name
        part = getproperty(parts, name)
        first_interval = get(event.intervals, name, nothing)
        second_interval = get(other.intervals, name, first_interval)
        first_interval === nothing && (first_interval = second_interval)
        first_interval === nothing && return massof(part)
        lower = max(first_interval.lower, second_interval.lower)
        upper = min(first_interval.upper, second_interval.upper)
        lower >= upper && return zero(float(lower))
        mass = smf(part, lower, upper)
        mass isa Real && isfinite(mass) && mass >= 0 ||
            throw(ArgumentError("parameter $name needs a scalar interval mass"))
        mass
    end
end

_evaluate(s::JointIntervalProbability, prior::NamedProductMeasure, ::Analytic, rng) =
    _estimate(_joint_interval_mass(prior, s.event))

function _evaluate(
    s::JointIntervalProbability,
    prior::MeasureBase.NormalizedMeasure{<:JointIntervalTiltKernel},
    ::Analytic,
    rng,
)
    kernel = prior.base
    score = _tilt_score(kernel)
    total = _joint_interval_mass(kernel.base, s.event)
    overlap = min(total, _joint_interval_mass(kernel.base, s.event, score.event))
    outside, inside = score.weights
    _estimate((outside*(total-overlap) + inside*overlap) / score.mass)
end
