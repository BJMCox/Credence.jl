"""`Expectation(f)` elicits the expectation of a pointwise function `f(x)`."""
struct Expectation{F} <: AbstractStatistic
    f::F
end
_moment_order(s::Expectation) = s.f === identity ? 1 : 0
function _evaluate(s::Expectation, prior, ::Analytic, rng)
    s.f === identity && return evaluate(Mean(), prior)
    throw(NoAnalytic(s))
end
_sample_estimate(s::Expectation, values, plan; joint=false) =
    _sample_estimate(Mean(), map(s.f, values), plan)

"""
    Quadrature(; rtol=1e-8, atol=1e-10, maxevals=100_000)

Adaptive scalar integration over the reference's full probability range.
Reported errors are estimates, rather than certified enclosures.
"""
struct Quadrature{R,A}
    rtol::R
    atol::A
    maxevals::Int
    function Quadrature(; rtol=1e-8, atol=1e-10, maxevals=100_000)
        isfinite(rtol) && rtol > 0 && isfinite(atol) && atol >= 0 && maxevals > 0 || throw(
            ArgumentError(
                "quadrature needs positive rtol and budget, and nonnegative atol",
            ),
        )
        new{typeof(rtol),typeof(atol)}(rtol, atol, maxevals)
    end
end

_tilt_feature(s::Expectation) = s.f

struct BoundedKLScore{F,P,B,M} <: AbstractTiltScore
    features::F
    parameters::P
    bounds::B
    mass::M
end
function (score::BoundedKLScore)(x)
    value =
        first(score.parameters) + sum(
            p * f(x) for (p, f) in zip(Base.tail(score.parameters), score.features);
            init=zero(first(score.parameters)),
        )
    clamp(value, log(score.bounds.lower), log(score.bounds.upper))
end
_tilt_envelope(score::BoundedKLScore) = score.bounds.upper
const UnknownTilt =
    MeasureBase.NormalizedMeasure{<:TiltKernel,<:MeasureBase.AbstractUnknownMass}
_analytic(s, prior::UnknownTilt, scope) = throw(NoAnalytic(s))
_analytic(s::On, prior::UnknownTilt, scope) = throw(NoAnalytic(s))
"""
    logdensity_estimate(report::TiltReport, x)

Evaluate normalized log density with the report's normalization evidence.
For numerical normalization, the returned bounds and evidence are estimated.
"""
function logdensity_estimate(report::TiltReport, x)
    prior = measure(report)
    if !(prior isa UnknownTilt)
        return Estimate(logdensityof(prior, x))
    end
    mass = report.normalization
    mass.lower > 0 || throw(ArgumentError("normalization uncertainty includes zero"))
    value = logdensityof(prior.base, x)
    Estimate(
        value - log(mass.value);
        lower=value-log(mass.upper),
        upper=value-log(mass.lower),
        evidence=mass.evidence,
    )
end

function _tilt_integrate(f, reference, features, plan; score=nothing)
    offset = smf(reference, -Inf)
    cuts = [zero(offset), one(offset)]
    for feature in features
        if feature isa IntervalEvent
            for x in (feature.lower, feature.upper)
                p = smf(reference, -Inf, x)
                0 < p < 1 && push!(cuts, p)
            end
        end
    end
    if score !== nothing && all(f -> f === identity, features)
        parameters = score.parameters
        slope = sum(Base.tail(parameters))
        if !iszero(slope)
            for logbound in (log(score.bounds.lower), log(score.bounds.upper))
                p = smf(reference, -Inf, (logbound-first(parameters))/slope)
                0 < p < 1 && push!(cuts, p)
            end
        end
    end
    value, error = quadgk(
        u -> f(invsmf(reference, offset+u)),
        sort!(unique!(cuts))...;
        rtol=plan.rtol,
        atol=plan.atol,
        maxevals=plan.maxevals,
        norm=x -> norm(x, Inf),
    )
    isfinite(error) && error <= max(plan.atol, plan.rtol*norm(value, Inf)) || throw(
        ErrorException("quadrature exhausted its budget before meeting the error target"),
    )
    value, error
end

function _tilt_data(reference, beliefs, method, plan)
    isone(massof(reference)) || throw(ArgumentError("reference must be normalized"))
    rootmeasure(reference) isa Union{MeasureBase.LebesgueBase,MeasureBase.Lebesgue} &&
    MeasureBase.mspace_elsize(reference) == () ||
        throw(ArgumentError("quadrature tilting requires a scalar continuous reference"))
    beliefs = _tilt_beliefs(beliefs)
    features = map(b -> _tilt_feature(b.statistic), beliefs)
    any(f -> f === identity, features) && _finite_mean(reference)
    moments, error =
        _tilt_integrate(x -> abs.([f(x) for f in features]), reference, features, plan)
    all(isfinite, moments) && isfinite(error) ||
        throw(ArgumentError("tilting features need finite absolute reference moments"))
    scales = map(eachindex(beliefs)) do i
        bounds = filter(isfinite, _bounds(beliefs[i].relation))
        scale = max(moments[i], maximum(abs, bounds; init=zero(moments[i])))
        iszero(scale) ? one(scale) : scale
    end
    terms = []
    for (i, belief) in enumerate(beliefs)
        lower, upper = _bounds(belief.relation)
        if lower == upper
            push!(terms, (; index=i, sign=1, target=lower/scales[i], lower=(-Inf)))
        else
            isfinite(lower) &&
                push!(terms, (; index=i, sign=1, target=lower/scales[i], lower=0.0))
            isfinite(upper) &&
                push!(terms, (; index=i, sign=-1, target=(-upper/scales[i]), lower=0.0))
        end
    end
    (; reference, beliefs, method, plan, features, scales, terms=Tuple(terms))
end

function _tilt_parameters(data, point)
    slopes = zeros(eltype(point), length(data.features))
    for (value, term) in zip(view(point, 2:length(point)), data.terms)
        slopes[term.index] += term.sign * value / data.scales[term.index]
    end
    (first(point), slopes...)
end

function _tilt_dual(data, point)
    parameters = _tilt_parameters(data, point)
    slopes = collect(Base.tail(parameters))
    link = BoundedKLScore(
        data.features,
        parameters,
        data.method,
        MeasureBase.UnknownFiniteMass(),
    )
    values, error =
        _tilt_integrate(data.reference, data.features, data.plan; score=link) do x
            features = [f(x) for f in data.features]
            score = first(parameters) + dot(slopes, features)
            logweight = clamp(score, log(data.method.lower), log(data.method.upper))
            weight = exp(logweight)
            conjugate = iszero(weight) ? -one(weight) : weight*(score-logweight+1)-1
            [conjugate; weight; weight .* features ./ data.scales]
        end
    objective =
        values[1] - first(point) - sum(
            point[j+1] * term.target for (j, term) in enumerate(data.terms);
            init=zero(first(point)),
        )
    gradient = [
        values[2]-1;
        [term.sign*values[2+term.index]-term.target for term in data.terms]
    ]
    objective, gradient, error
end

_tilt_mass(reference, score) = MeasureBase.UnknownFiniteMass()
const UniformTiltKernel = MeasureBase.DensityMeasure{
    <:MeasureBase.LogFuncDensity{<:BoundedKLScore{<:Any,<:Any,<:Any,<:Real}},
    <:MeasureBase.StdUniform,
}

function _uniform_tilt_cuts(score, endpoint=1)
    a, b = first(score.parameters), sum(Base.tail(score.parameters))
    cuts = [zero(a), oftype(a, endpoint)]
    if !iszero(b)
        for bound in (score.bounds.lower, score.bounds.upper)
            x = (log(bound)-a)/b
            0 < x < endpoint && push!(cuts, x)
        end
    end
    sort!(unique!(cuts))
end

# The affine score has an analytic integral on each unclipped or constant segment.
function _uniform_tilt_integral(score, endpoint=1; moment=false)
    a, b = first(score.parameters), sum(Base.tail(score.parameters))
    cuts = _uniform_tilt_cuts(score, endpoint)
    total = zero(a)
    for (c, d) in zip(cuts[1:(end-1)], cuts[2:end])
        midpoint = a+b*(c+d)/2
        logweight = clamp(midpoint, log(score.bounds.lower), log(score.bounds.upper))
        if iszero(b) || midpoint != logweight
            total += moment ? exp(logweight)*(d*d-c*c)/2 : exp(logweight)*(d-c)
        else
            delta = b*(d-c)
            left, right = exp(a+b*c), exp(a+b*d)
            mass = if iszero(delta)
                left*(d-c)
            elseif abs(delta) < 0.5
                left*(d-c)*expm1(delta)/delta
            else
                (right-left)/b
            end
            if moment && abs(delta) < 0.5
                coefficient = one(delta)
                series = zero(delta)
                k = 0
                while true
                    updated = series + coefficient/(k+2)
                    updated == series && break
                    series = updated
                    k += 1
                    coefficient *= delta/k
                end
                total += c*mass + left*(d-c)^2*series
            elseif moment
                total += (d*right-c*left-mass)/b
            else
                total += mass
            end
        end
    end
    total
end
function _tilt_mass(reference::MeasureBase.StdUniform, score)
    all(f -> f === identity, score.features) || return MeasureBase.UnknownFiniteMass()
    _uniform_tilt_integral(score)
end
MeasureBase.smf(kernel::UniformTiltKernel, x) =
    _uniform_tilt_integral(_tilt_score(kernel), clamp(x, 0, 1))
MeasureBase.smf(kernel::UniformTiltKernel, a, b) = smf(kernel, b)-smf(kernel, a)
function MeasureBase.invsmf(kernel::UniformTiltKernel, p)
    score = _tilt_score(kernel)
    0 <= p <= score.mass || return oftype(float(p), NaN)
    cuts = _uniform_tilt_cuts(score)
    a, b = first(score.parameters), sum(Base.tail(score.parameters))
    cumulative = zero(p)
    for (c, d) in zip(cuts[1:(end-1)], cuts[2:end])
        next = _uniform_tilt_integral(score, d)
        if p <= next && next > cumulative
            midpoint = a+b*(c+d)/2
            logweight = clamp(midpoint, log(score.bounds.lower), log(score.bounds.upper))
            value = if iszero(b) || midpoint != logweight
                c+(p-cumulative)/exp(logweight)
            else
                c+log1p(b*(p-cumulative)/exp(a+b*c))/b
            end
            return clamp(value, c, d)
        end
        cumulative = next
    end
    one(p)
end
Statistics.mean(prior::MeasureBase.NormalizedMeasure{<:UniformTiltKernel}) =
    _uniform_tilt_integral(_tilt_score(prior.base); moment=true) / massof(prior.base)
_evaluate(
    s::Std,
    prior::MeasureBase.NormalizedMeasure{<:UniformTiltKernel},
    ::Analytic,
    rng,
) = throw(NoAnalytic(s))

const UniformEventKernel = MeasureBase.DensityMeasure{
    <:MeasureBase.LogFuncDensity{<:EventTiltScore},
    <:MeasureBase.StdUniform,
}
function Statistics.mean(prior::MeasureBase.NormalizedMeasure{<:UniformEventKernel})
    score = _tilt_score(prior.base)
    a, b = clamp(score.event.lower, 0, 1), clamp(score.event.upper, 0, 1)
    (score.weights[1]/2 + (score.weights[2]-score.weights[1])*(b*b-a*a)/2) / score.mass
end
_evaluate(
    s::Mean,
    prior::MeasureBase.NormalizedMeasure{<:UniformEventKernel},
    ::Analytic,
    rng,
) = _estimate(mean(prior))
_boundary_tilt(reference, beliefs, method) = nothing
function _boundary_tilt(reference::MeasureBase.StdUniform, beliefs, method)
    length(beliefs) == 1 && _tilt_feature(only(beliefs).statistic) === identity ||
        return nothing
    method.lower == method.upper && return nothing
    fraction = (1-method.lower)/(method.upper-method.lower)
    least = method.lower/2 + (method.upper-method.lower)*fraction^2/2
    greatest = 1-least
    lower, upper = _bounds(only(beliefs).relation)
    max(lower, least) <= min(upper, greatest) ||
        throw(ArgumentError("mean is infeasible under the ratio bounds"))
    target = clamp(0.5, lower, upper)
    if target == greatest
        score = EventTiltScore(
            IntervalEvent(1-fraction, Inf),
            (method.lower, method.upper),
            one(target),
        )
    elseif target == least
        score = EventTiltScore(
            IntervalEvent(-Inf, fraction),
            (method.lower, method.upper),
            one(target),
        )
    else
        return nothing
    end
    prior = normalize(mintegrate_exp(score, reference))
    divergence = sum(zip((1-fraction, fraction), score.weights)) do (p, r)
        iszero(r) ? zero(r) : p*r*log(r)
    end
    _tilt_report(
        reference,
        beliefs,
        method,
        prior,
        (; retcode=:Analytic),
        Estimate(1.0),
        Estimate(divergence),
        extrema(score.weights),
        :feasible,
        :analytic,
    )
end

function _tilt_continuous_report(data, point, solver; atol, converged)
    parameters = _tilt_parameters(data, point)
    initial = BoundedKLScore(
        data.features,
        parameters,
        data.method,
        MeasureBase.UnknownFiniteMass(),
    )
    knownmass = _tilt_mass(data.reference, initial)
    score = BoundedKLScore(data.features, parameters, data.method, knownmass)
    values, error = _tilt_integrate(data.reference, data.features, data.plan; score) do x
        logweight = score(x)
        weight = exp(logweight)
        entropy = iszero(weight) ? zero(weight) : weight*logweight
        [
            weight;
            entropy;
            [weight*f(x)/scale for (f, scale) in zip(data.features, data.scales)]
        ]
    end
    mass = if knownmass isa Real
        Estimate(knownmass)
    else
        Estimate(
            values[1];
            lower=max(0, values[1]-error),
            upper=values[1]+error,
            evidence=:estimated,
        )
    end
    mass.lower > 0 || throw(ArgumentError("normalization estimate includes zero"))
    ratios = if knownmass isa Real
        low, high = extrema((exp(score(0)), exp(score(1))))
        (low/mass.value, high/mass.value)
    else
        (data.method.lower/mass.upper, data.method.upper/mass.lower)
    end
    kernel = mintegrate_exp(score, data.reference)
    prior = normalize(kernel)
    residual = maximum(eachindex(data.beliefs); init=zero(mass.value)) do i
        value = values[i+2]*data.scales[i]/mass.value
        lower, upper = _bounds(data.beliefs[i].relation)
        max(lower-value, value-upper, zero(value))/data.scales[i]
    end
    divergence = values[2]/mass.value-log(mass.value)
    candidates =
        [b/z for b in (values[2]-error, values[2]+error), z in (mass.lower, mass.upper)]
    estimate = Estimate(
        divergence;
        lower=minimum(candidates)-log(mass.upper),
        upper=maximum(candidates)-log(mass.lower),
        evidence=:estimated,
    )
    feasibility = if knownmass isa Real && residual <= atol
        checks =
            map(_check_estimate, data.beliefs, _estimates(data.beliefs, prior, nothing))
        if all(c -> c.status == :satisfied, checks) &&
           data.method.lower <= ratios[1] <= ratios[2] <= data.method.upper
            :feasible
        else
            :within_tolerance
        end
    elseif residual <= atol
        :estimated
    else
        :unresolved
    end
    _, gradient, gradient_error = _tilt_dual(data, point)
    stationary =
        gradient_error <= atol &&
        abs(first(gradient)) <= atol &&
        all(enumerate(data.terms)) do (j, term)
            if term.lower == 0 && point[j+1] <= atol
                gradient[j+1] >= -atol
            else
                abs(gradient[j+1]) <= atol
            end
        end
    optimality = converged && stationary ? :numerical : :unresolved
    _tilt_report(
        data.reference,
        data.beliefs,
        data.method,
        prior,
        solver,
        mass,
        estimate,
        ratios,
        feasibility,
        optimality,
    )
end
