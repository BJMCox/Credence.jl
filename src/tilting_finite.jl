"""
    FiniteReference(points, weights)

Declare an actual finite reference law. Weights are normalized before fitting.
Continuous integration points are specified through `Quadrature` instead.
"""
struct FiniteReference{P,W}
    points::P
    weights::W
    function FiniteReference(points, weights)
        points, weights = Tuple(points), collect(float.(weights))
        !isempty(points) && length(points) == length(weights) || throw(
            DimensionMismatch(
                "finite points and weights must have the same positive length",
            ),
        )
        for i in eachindex(points)
            points[i] == points[i] && all(j -> points[i] != points[j], 1:(i-1)) ||
                throw(ArgumentError("finite reference points must define distinct atoms"))
        end
        all(w -> isfinite(w) && w >= 0, weights) && maximum(weights) > 0 || throw(
            ArgumentError("finite weights must be nonnegative with positive total mass"),
        )
        weights ./= maximum(weights)
        weights ./= sum(weights)
        new{typeof(points),typeof(weights)}(points, weights)
    end
end
function _finite_law(points, weights)
    kept = filter(pair -> !iszero(last(pair)), Tuple(zip(points, weights)))
    points = first.(kept)
    components = map(pair -> last(pair)*Dirac(first(pair)), kept)
    masses = collect(map(massof, components))
    masses ./= sum(masses)
    base = normalize(superpose(components))
    normalize(mintegrate_exp(FiniteTiltScore(points, masses), base))
end
measure(reference::FiniteReference) = _finite_law(reference.points, reference.weights)

struct FiniteTiltScore{P,W} <: AbstractTiltScore
    points::P
    weights::W
end
(score::FiniteTiltScore)(x) = zero(eltype(score.weights))
const FiniteTiltKernel =
    MeasureBase.DensityMeasure{<:MeasureBase.LogFuncDensity{<:FiniteTiltScore}}
const FiniteTiltLaw = MeasureBase.NormalizedMeasure{<:FiniteTiltKernel}
MeasureBase.massof(kernel::FiniteTiltKernel) = one(eltype(_tilt_score(kernel).weights))
MeasureBase.rand_impl(ctx::GenContext, kernel::FiniteTiltKernel) = rand(ctx, kernel.base)
_check_moment(statistic, prior::FiniteTiltLaw, scope) = nothing
function _finite_expectation(feature, prior::FiniteTiltLaw)
    score = _tilt_score(prior.base)
    sum(
        weight*feature(x) for
        (x, weight) in zip(score.points, score.weights) if !iszero(weight)
    )
end
_evaluate(s::Mean, prior::FiniteTiltLaw, ::Analytic, rng) =
    _estimate(_finite_expectation(identity, prior))
_evaluate(s::Expectation, prior::FiniteTiltLaw, ::Analytic, rng) =
    _estimate(_finite_expectation(s.f, prior))
_evaluate(s::Probability, prior::FiniteTiltLaw, ::Analytic, rng) =
    _estimate(_finite_expectation(s.event, prior))
_evaluate(s::Probability{<:IntervalEvent}, prior::FiniteTiltLaw, ::Analytic, rng) =
    _estimate(_finite_expectation(s.event, prior))
_analytic(s::On, prior::FiniteTiltLaw, scope) =
    _estimate(_finite_expectation(_tilt_feature(s), prior))
Statistics.mean(prior::FiniteTiltLaw) = _finite_expectation(identity, prior)
function Statistics.std(prior::FiniteTiltLaw)
    center = mean(prior)
    sqrt(_finite_expectation(x -> abs2(x-center), prior))
end
function MeasureBase.smf(kernel::FiniteTiltKernel, x)
    score = _tilt_score(kernel)
    sum(w for (point, w) in zip(score.points, score.weights) if point <= x; init=zero(x))
end
MeasureBase.smf(kernel::FiniteTiltKernel, a, b) = smf(kernel, b)-smf(kernel, a)
function MeasureBase.invsmf(kernel::FiniteTiltKernel, p)
    score = _tilt_score(kernel)
    cumulative = zero(p)
    for i in sortperm(collect(score.points))
        iszero(score.weights[i]) && continue
        cumulative += score.weights[i]
        cumulative >= p && return score.points[i]
    end
    maximum(score.points)
end
