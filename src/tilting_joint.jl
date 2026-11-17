const NamedProductMeasure = MeasureBase.ProductMeasure{<:NamedTuple}
const JointIntervalProbability = Probability{<:JointIntervalEvent}
const JointIntervalTiltKernel = MeasureBase.DensityMeasure{
    <:MeasureBase.LogFuncDensity{<:EventTiltScore{<:JointIntervalEvent}},
}

function _interval_mass(part, lower, upper)
    lower >= upper && return zero(float(lower))
    mass = smf(part, lower, upper)
    mass isa Real && isfinite(mass) && mass >= 0 ||
        throw(ArgumentError("joint probability needs scalar interval masses"))
    mass
end

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
        _interval_mass(part, lower, upper)
    end
end

struct FixedMarginalTiltScore{E,W,P,T} <: AbstractTiltScore
    event::E
    weights::W
    probabilities::P
    mass::T
end
function (score::FixedMarginalTiltScore)(x)
    a, b = map(keys(score.event.intervals), values(score.event.intervals)) do name, interval
        interval(getproperty(x, name))
    end
    log(score.weights[1+Int(a)+2*Int(b)])
end
_tilt_envelope(score::FixedMarginalTiltScore) = maximum(score.weights)
const FixedMarginalTiltKernel =
    MeasureBase.DensityMeasure{<:MeasureBase.LogFuncDensity{<:FixedMarginalTiltScore}}
const NormalizedFixedMarginalTilt = MeasureBase.NormalizedMeasure{<:FixedMarginalTiltKernel}
MeasureBase.marginals(prior::NormalizedFixedMarginalTilt) = marginals(prior.base.base)

function _interval_partition(part, event, query=IntervalEvent(-Inf, Inf))
    a, b = query.lower, query.upper
    inside = _interval_mass(part, max(a, event.lower), min(b, event.upper))
    outside =
        _interval_mass(part, a, min(b, event.lower)) +
        _interval_mass(part, max(a, event.upper), b)
    outside, inside
end

function _fixed_marginal_tilt(reference, beliefs, method)
    reference isa NamedProductMeasure &&
    length(beliefs) == 1 &&
    only(beliefs).statistic isa JointIntervalProbability || throw(
        ArgumentError(
            "preserve_marginals needs one joint probability belief and a named independent product",
        ),
    )
    belief = only(beliefs)
    event = belief.statistic.event
    parts = marginals(reference)
    names = keys(event.intervals)
    length(names) == 2 && all(name -> haskey(parts, name), names) || throw(
        ArgumentError(
            "preserve_marginals needs intervals for two named reference parameters",
        ),
    )
    all(part -> isone(massof(part)), parts) ||
        throw(ArgumentError("reference marginals must have unit mass"))

    (a0, a1), (b0, b1) = map(names, values(event.intervals)) do name, interval
        _interval_partition(parts[name], interval)
    end
    p = a1*b1
    lower, upper = _bounds(belief.relation)
    target = clamp(p, lower, upper)
    target == p && return _reference_tilt(reference, beliefs, method)

    # Cell order: neither, first only, second only, both. Fixed row and column sums
    # leave one free change, added to matching cells and subtracted from mixed cells.
    reference_cells = (a0*b0, a1*b0, a0*b1, p)
    signs = (1, -1, -1, 1)
    limits = map(reference_cells, signs) do cell, sign
        minmax(sign*(method.lower*cell-cell), sign*(method.upper*cell-cell))
    end
    lo, hi = maximum(first, limits), minimum(last, limits)
    minimum_target = max(p+lo, method.lower*p)
    maximum_target = min(p+hi, method.upper*p)
    minimum_target <= target <= maximum_target || throw(
        ArgumentError(
            "joint probability is infeasible with fixed marginals. Attainable range: $minimum_target to $maximum_target",
        ),
    )
    delta = clamp(target-p, lo, hi)
    cells = map(reference_cells, signs) do cell, sign
        clamp(cell+sign*delta, method.lower*cell, method.upper*cell)
    end
    cells = (cells[1], cells[2], cells[3], target)
    weights = clamp.(cells ./ reference_cells, method.lower, method.upper)
    score = FixedMarginalTiltScore(event, weights, cells, one(p))
    prior = normalize(mintegrate_exp(score, reference))
    divergence = sum(zip(cells, weights)) do (cell, weight)
        iszero(cell) ? zero(cell) : cell*log(weight)
    end
    _tilt_report(
        reference,
        beliefs,
        method,
        prior,
        (; retcode=:Analytic),
        Estimate(score.mass),
        Estimate(max(zero(divergence), divergence)),
        extrema(weights),
        :feasible,
        :analytic,
    )
end

function _evaluate(
    s::JointIntervalProbability,
    prior::NormalizedFixedMarginalTilt,
    ::Analytic,
    rng,
)
    kernel = prior.base
    reference = kernel.base
    score = _tilt_score(kernel)
    query = s.event.intervals
    names = keys(score.event.intervals)
    # A query involving at most one tilted factor has the reference probability.
    if !all(name -> haskey(query, name), names)
        return _estimate(_joint_interval_mass(reference, s.event))
    end
    parts = marginals(reference)
    (a0, a1), (b0, b1) = map(names, values(score.event.intervals)) do name, interval
        part = parts[name]
        _interval_partition(part, interval, query[name]) ./
        _interval_partition(part, interval)
    end
    fractions = (a0*b0, a1*b0, a0*b1, a1*b1)
    other = Base.structdiff(query, score.event.intervals)
    mass = sum(score.probabilities .* fractions)
    isempty(other) || (mass *= _joint_interval_mass(reference, JointIntervalEvent(other)))
    _estimate(clamp(mass, 0, 1))
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
