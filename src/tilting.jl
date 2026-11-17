"""
    BoundedKL(; lower, upper)

Minimize KL divergence to a fixed reference subject to beliefs and bounds
`lower <= dQ/dP <= upper`. The bounds satisfy `0 <= lower <= 1 <= upper < Inf`.
"""
struct BoundedKL{L,U}
    lower::L
    upper::U
    function BoundedKL(; lower::Real, upper::Real)
        0 <= lower <= 1 <= upper < Inf ||
            throw(ArgumentError("ratio bounds must satisfy 0 <= lower <= 1 <= upper < Inf"))
        new{typeof(lower),typeof(upper)}(lower, upper)
    end
end

"""
    TiltReport

A fixed-reference projection with its fit, normalization, divergence, ratio bounds,
feasibility and optimality evidence. `measure(report)` returns the fitted law.
"""
struct TiltReport{P,M,F,N,D,R}
    reference::P
    method::M
    fit::F
    normalization::N
    divergence::D
    ratio_bounds::R
    feasibility::Symbol
    optimality::Symbol
end
_fit_report(report::TiltReport) = report.fit
function measure(report::TiltReport)
    result = only(report.fit.results)
    result.measure === nothing && throw(ArgumentError("tilting did not produce a measure"))
    result.measure
end
function validate(rng::Union{AbstractRNG,GenContext}, report::TiltReport; kwargs...)
    validation = validate(rng, report.fit; kwargs...)
    Validation(report, validation.checks, validation.plan, validation.seed)
end
function select(report::TiltReport, result_id=(:tilt, 1); role=:accepted, kwargs...)
    role == :accepted &&
        report.feasibility != :feasible &&
        throw(ArgumentError("projection feasibility remains unresolved"))
    role == :accepted &&
        !(report.optimality in (:analytic, :numerical)) &&
        throw(ArgumentError("projection optimality remains unresolved"))
    _select(report, result_id; role, kwargs...)
end
fit_summary(report::TiltReport) = fit_summary(report.fit)
function _tilt_report(
    reference,
    beliefs,
    method,
    prior,
    solver,
    normalization,
    divergence,
    ratio_bounds,
    feasibility,
    optimality,
)
    candidate = Candidate(:tilt, Returns(prior), ((),))
    problem = Elicitation(beliefs, (candidate,))
    error =
        prior === nothing ? ErrorException("tilting did not produce a measure") : nothing
    result = FitResult((:tilt, 1), candidate, (), prior, solver, error)
    fit = FitReport(problem, (result,))
    TiltReport(
        reference,
        method,
        fit,
        normalization,
        divergence,
        ratio_bounds,
        feasibility,
        optimality,
    )
end

_tilt_feature(::Mean) = identity
_tilt_feature(s::Probability) = s.event
function _tilt_feature(s::On)
    s.target isa Predictive &&
        throw(ArgumentError("predictive tilting requires conditional expectations"))
    feature = _tilt_feature(s.statistic)
    x -> feature(_draw_target(x, nothing, s.target))
end
function _tilt_beliefs(beliefs)
    beliefs = Tuple(beliefs)
    all(b -> b isa Hard, beliefs) ||
        throw(ArgumentError("bounded KL requires hard linear expectation beliefs"))
    for belief in beliefs
        _tilt_feature(belief.statistic)
        lower, upper = _bounds(belief.relation)
        lower isa Real && upper isa Real ||
            throw(ArgumentError("tilting requires scalar expectation bounds"))
    end
    beliefs
end

function _reference_tilt(reference, beliefs, method)
    isone(massof(reference)) || throw(ArgumentError("reference must be normalized"))
    checks = map(_check_estimate, beliefs, _estimates(beliefs, reference, nothing))
    satisfied = all(c -> c.status == :satisfied, checks)
    fixed = method.lower == 1 || method.upper == 1
    if fixed && any(c -> c.status == :violated, checks)
        throw(ArgumentError("ratio bounds force the reference, which violates a belief"))
    end
    satisfied || fixed || return nothing
    _tilt_report(
        reference,
        beliefs,
        method,
        reference,
        (; retcode=:Analytic),
        Estimate(1.0),
        Estimate(0.0),
        (1.0, 1.0),
        satisfied ? :feasible : :unresolved,
        :analytic,
    )
end

abstract type AbstractTiltScore end
struct EventTiltScore{F,W,M} <: AbstractTiltScore
    event::F
    weights::W
    mass::M
end
(score::EventTiltScore)(x) = log(score.weights[1+Int(score.event(x))])
_tilt_envelope(score::EventTiltScore) = maximum(score.weights)
const TiltKernel =
    MeasureBase.DensityMeasure{<:MeasureBase.LogFuncDensity{<:AbstractTiltScore}}
const EventTiltKernel =
    MeasureBase.DensityMeasure{<:MeasureBase.LogFuncDensity{<:EventTiltScore}}
const IntervalTiltKernel = MeasureBase.DensityMeasure{
    <:MeasureBase.LogFuncDensity{<:EventTiltScore{<:IntervalEvent}},
}
const NormalizedTilt = MeasureBase.NormalizedMeasure{<:TiltKernel}
_tilt_score(kernel::TiltKernel) = MeasureBase.logdensityof(kernel.f)
MeasureBase.massof(kernel::TiltKernel) = _tilt_score(kernel).mass
function MeasureBase.rand_impl(ctx::GenContext, kernel::TiltKernel)
    score = _tilt_score(kernel)
    envelope = log(_tilt_envelope(score))
    while true
        x = rand(ctx, kernel.base)
        log(rand(ctx)) < score(x) - envelope && return x
    end
end
function _check_moment(statistic, prior::NormalizedTilt, scope)
    _moment_order(_statistic(statistic)) == 0 && return nothing
    score = _tilt_score(prior.base)
    # A bounded event can remove divergent reference tails entirely.
    if score isa EventTiltScore{<:IntervalEvent} &&
       iszero(first(score.weights)) &&
       isfinite(score.event.lower) &&
       isfinite(score.event.upper) &&
       (
           statistic isa Union{Mean,Std} ||
           (statistic isa Expectation && statistic.f === identity)
       )
        return nothing
    end
    reference = prior.base.base
    try
        _check_moment(statistic, reference, scope)
        _analytic(statistic, reference, scope)
    catch error
        error isa Union{NoAnalytic,MethodError} || rethrow()
        throw(ArgumentError("tilted moment finiteness is unresolved"))
    end
    nothing
end
_evaluate(
    s::Mean,
    prior::MeasureBase.NormalizedMeasure{<:EventTiltKernel},
    ::Analytic,
    rng,
) = throw(NoAnalytic(s))
_evaluate(
    s::Std,
    prior::MeasureBase.NormalizedMeasure{<:EventTiltKernel},
    ::Analytic,
    rng,
) = throw(NoAnalytic(s))
function MeasureBase.smf(kernel::IntervalTiltKernel, x)
    score = _tilt_score(kernel)
    event = score.event
    total = smf(kernel.base, -Inf, x)
    selected =
        x <= event.lower ? zero(total) : smf(kernel.base, event.lower, min(x, event.upper))
    score.weights[1] * total + (score.weights[2]-score.weights[1]) * selected
end
MeasureBase.smf(kernel::IntervalTiltKernel, a, b) = smf(kernel, b) - smf(kernel, a)
function MeasureBase.invsmf(kernel::IntervalTiltKernel, p)
    score = _tilt_score(kernel)
    a = smf(kernel.base, -Inf, score.event.lower)
    b = smf(kernel.base, -Inf, score.event.upper)
    w0, w1 = score.weights
    u = if p <= w0*a && w0 > 0
        p/w0
    elseif p <= w0*a + w1*(b-a) && w1 > 0
        a + (p-w0*a)/w1
    else
        b + (p-w0*a-w1*(b-a))/w0
    end
    invsmf(kernel.base, smf(kernel.base, -Inf) + u)
end

"""
    tilt(reference, beliefs, method::BoundedKL, algorithm; kwargs...)
    tilt(reference, beliefs, method::BoundedKL; preserve_marginals=false)

Project a fixed probability reference onto linear expectation requirements.
An Optimization.jl algorithm fits general constraints. A single interval-probability
constraint has an analytic two-cell solution and requires no optimizer. Use named
intervals in `Probability` for a joint update of an independent product. Set
`preserve_marginals=true` to keep both marginal laws when a single joint probability
belief names both parameters of a two-factor reference.
"""
function tilt(reference, beliefs, method::BoundedKL; preserve_marginals=false)
    beliefs = _tilt_beliefs(beliefs)
    preserve_marginals && return _fixed_marginal_tilt(reference, beliefs, method)
    report = _event_tilt(reference, beliefs, method)
    report === nothing && (report = _reference_tilt(reference, beliefs, method))
    report === nothing && (report = _boundary_tilt(reference, beliefs, method))
    report === nothing &&
        throw(ArgumentError("supply an Optimization.jl algorithm for these beliefs"))
    report
end
function _event_tilt(reference, beliefs, method)
    length(beliefs) == 1 || return nothing
    belief = only(beliefs)
    statistic = belief.statistic
    statistic isa Probability{<:Union{IntervalEvent,JointIntervalEvent}} || return nothing
    if statistic isa JointIntervalProbability && !(reference isa NamedProductMeasure)
        throw(ArgumentError("joint probability tilting needs a named independent product"))
    end
    isone(massof(reference)) || throw(ArgumentError("reference must be normalized"))
    p = evaluate(statistic, reference).value
    lo = max(method.lower*p, 1-method.upper*(1-p))
    hi = min(method.upper*p, 1-method.lower*(1-p))
    lower, upper = _bounds(belief.relation)
    max(lo, lower) <= min(hi, upper) ||
        throw(ArgumentError("event probability is infeasible under the ratio bounds"))
    target = clamp(p, lower, upper)
    target == p && return _reference_tilt(reference, beliefs, method)
    weights = (p == 1 ? one(p) : (1-target)/(1-p), p == 0 ? one(p) : target/p)
    mass = (1-p)*weights[1] + p*weights[2]
    score = EventTiltScore(statistic.event, weights, mass)
    prior = normalize(mintegrate_exp(score, reference))
    divergence =
        sum(zip((1-p, p), weights)) do (probability, weight)
            iszero(weight) ? zero(weight) : probability * weight * log(weight)
        end / mass - log(mass)
    solver = (; retcode=:Analytic)
    _tilt_report(
        reference,
        beliefs,
        method,
        prior,
        solver,
        Estimate(mass),
        Estimate(divergence),
        extrema(weights ./ mass),
        :feasible,
        :analytic,
    )
end
