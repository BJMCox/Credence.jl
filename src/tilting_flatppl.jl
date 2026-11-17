function _flatppl(prior::FiniteTiltKernel, path)
    score = _tilt_score(prior)
    parts = (
        _flatppl_block(
            "weighted(",
            (_flatppl_value(weight, path), "Dirac(" * _flatppl_value(point, path) * ")"),
            ")",
        ) for (point, weight) in zip(score.points, score.weights) if !iszero(weight)
    )
    _flatppl_block("superpose(", parts, ")")
end

function _flatppl(prior::EventTiltKernel, path)
    score = _tilt_score(prior)
    base = _flatppl(prior.base, path)
    condition = _flatppl_event(score.event, "x", path)
    outside, inside = map(w -> _flatppl_value(w, path), score.weights)
    callback = "x -> ifelse($condition, $inside, $outside)"
    _flatppl_block("weighted(", (callback, base), ")")
end

function _flatppl_event(event::IntervalEvent, variable, path)
    lower, upper = _flatppl_value(event.lower, path), _flatppl_value(event.upper, path)
    "($lower < $variable) && ($variable <= $upper)"
end
function _flatppl_event(event::JointIntervalEvent, variable, path)
    join(
        (
            _flatppl_event(interval, "$variable.$name", "$path.$name") for
            (name, interval) in pairs(event.intervals)
        ),
        " && ",
    )
end

function _flatppl(prior::UniformTiltKernel, path)
    score = _tilt_score(prior)
    all(f -> f === identity, score.features) || return _flatppl_opaque_tilt(prior, path)
    intercept = _flatppl_value(first(score.parameters), path)
    slope = _flatppl_value(sum(Base.tail(score.parameters)), path)
    lower, upper =
        map(w -> _flatppl_value(log(w), path), (score.bounds.lower, score.bounds.upper))
    callback =
        "x -> min(max(" * intercept * " + " * slope * " * x, " * lower * "), " * upper * ")"
    _flatppl_block("logweighted(", (callback, _flatppl(prior.base, path)), ")")
end

_flatppl(prior::TiltKernel, path) = _flatppl_opaque_tilt(prior, path)
_flatppl_opaque_tilt(prior, path) =
    throw(ArgumentError("opaque tilting feature at $path has no FlatPPL translation"))

# Preserve the known normalizer rather than asking the target to estimate it again.
function _flatppl(prior::NormalizedTilt, path)
    prior.logmass isa Real || throw(ArgumentError("unknown tilt normalization at $path"))
    _flatppl_block(
        "weighted(",
        (_flatppl_value(exp(-prior.logmass), path), _flatppl(prior.base, path)),
        ")",
    )
end
