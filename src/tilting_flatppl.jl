function _flatppl(prior::EventTiltKernel, path)
    score = _tilt_score(prior)
    event = score.event
    event isa IntervalEvent || return _flatppl_opaque_tilt(prior, path)
    lower, upper = _flatppl_value(event.lower, path), _flatppl_value(event.upper, path)
    outside, inside = map(w -> _flatppl_value(w, path), score.weights)
    callback =
        "x -> ifelse((" *
        lower *
        " < x) && (x <= " *
        upper *
        "), " *
        inside *
        ", " *
        outside *
        ")"
    _flatppl_block("weighted(", (callback, _flatppl(prior.base, path)), ")")
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
