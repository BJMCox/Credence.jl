abstract type AbstractStatistic end

"""
    Estimate(value; lower=value, upper=value, evidence=:analytic)

An evaluated statistic, its enclosure, and the source of its evidence.
"""
struct Estimate{V,L,U}
    value::V
    lower::L
    upper::U
    evidence::Symbol
    n::Int
    function Estimate(value; lower=value, upper=value, evidence=:analytic, n=0)
        all(lower .<= value) && all(value .<= upper) ||
            throw(ArgumentError("estimate lies outside its enclosure"))
        new{typeof(value),typeof(lower),typeof(upper)}(value, lower, upper, evidence, n)
    end
end

struct IntervalEvent{A,B}
    lower::A
    upper::B
end
(event::IntervalEvent)(x) = event.lower < x <= event.upper

"""
    Probability(a, b)
    Probability(predicate)

The probability P(a < X <= b), or an event defined by a predicate.
A predicate requires a sampling evaluation plan.
"""
struct Probability{F} <: AbstractStatistic
    event::F
end
function Probability(lower::Real, upper::Real)
    lower < upper || throw(ArgumentError("interval endpoints must increase"))
    Probability(IntervalEvent(lower, upper))
end

"""
    Quantile(p)

The lower quantile at probability p, with 0 < p < 1.
"""
struct Quantile{T} <: AbstractStatistic
    p::T
    function Quantile(p::Real)
        0 < p < 1 || throw(ArgumentError("quantile probability must lie in (0,1)"))
        new{typeof(p)}(p)
    end
end

"""
    Mean()

The population mean, when it exists.
"""
struct Mean <: AbstractStatistic end
"""
    Std()

The population standard deviation, when it exists.
"""
struct Std <: AbstractStatistic end

"""The covariance of two scalar quantities selected by `Joint`."""
struct Covariance <: AbstractStatistic end
"""The correlation of two scalar quantities selected by `Joint`."""
struct Correlation <: AbstractStatistic end
"""The covariance matrix of a vector-valued quantity or a `Joint` selection."""
struct CovarianceMatrix <: AbstractStatistic end
"""The correlation matrix of a vector-valued quantity or a `Joint` selection."""
struct CorrelationMatrix <: AbstractStatistic end

"""
    Statistic(f; label=:custom)

A user-supplied analytic statistic `f(measure)`, optionally returning an `Estimate`.
"""
struct Statistic{F,L} <: AbstractStatistic
    f::F
    label::L
end
Statistic(f; label=:custom) = Statistic(f, label)

"""
    Analytic()

Evaluate through analytic measure operations.
"""
struct Analytic end

"""
    MonteCarlo(n; alpha=0.01)

Evaluate statistics from n independent draws with numerical uncertainty at level alpha.
"""
struct MonteCarlo{T}
    n::Int
    alpha::T
    function MonteCarlo(n::Integer; alpha=0.01)
        n > 0 || throw(ArgumentError("sample count must be positive"))
        0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0,1)"))
        new{typeof(alpha)}(n, alpha)
    end
end

"""
    evaluate(statistic, prior; plan=Analytic(), rng=nothing)

Evaluate a statistic under a normalized MeasureBase measure.
"""
function evaluate(statistic::AbstractStatistic, prior; plan=Analytic(), rng=nothing)
    isone(MeasureBase.massof(prior)) || throw(ArgumentError("candidate must be normalized"))
    _evaluate(statistic, prior, plan, rng)
end

function _estimate(x::Estimate)
    all(isfinite, x.value) || throw(ArgumentError("statistic is not finite"))
    x
end
function _estimate(x::AbstractArray)
    all(isfinite, x) || throw(ArgumentError("statistic is not finite"))
    Estimate(x)
end
function _estimate(x::Real)
    isfinite(x) || throw(ArgumentError("statistic is not finite"))
    Estimate(x)
end
_evaluate(s::Mean, prior, ::Analytic, rng) = _estimate(mean(prior))
_evaluate(s::Std, prior, ::Analytic, rng) = _estimate(std(prior))
function _finite_mean(prior)
    value = mean(prior)
    _estimate(_flatten(value))
    value
end
_covariance(prior) = cov(prior)
_covariance(prior::ProductMeasure{<:AbstractArray}) =
    Diagonal(vec(map(var, marginals(prior))))
_evaluate(::CovarianceMatrix, prior, ::Analytic, rng) = _estimate(_covariance(prior))
function _evaluate(::CorrelationMatrix, prior, ::Analytic, rng)
    covariance = _covariance(prior)
    scales = sqrt.(diag(covariance))
    _estimate(covariance ./ (scales * scales'))
end
function _evaluate(::Covariance, prior, ::Analytic, rng)
    covariance = _covariance(prior)
    size(covariance) == (2, 2) ||
        throw(DimensionMismatch("Covariance requires two scalar quantities"))
    # Independence gives zero cross-covariance only when the selected means exist.
    prior isa ProductMeasure && _finite_mean(prior)
    _estimate(covariance[1, 2])
end
function _evaluate(::Correlation, prior, ::Analytic, rng)
    covariance = _covariance(prior)
    size(covariance) == (2, 2) ||
        throw(DimensionMismatch("Correlation requires two scalar quantities"))
    first_scale = _estimate(sqrt(covariance[1, 1])).value
    second_scale = _estimate(sqrt(covariance[2, 2])).value
    _estimate(covariance[1, 2] / first_scale / second_scale)
end
function _evaluate(s::Quantile, prior, ::Analytic, rng)
    _estimate(MeasureBase.invsmf(prior, MeasureBase.smf(prior, -Inf) + s.p))
end
_evaluate(s::Quantile, prior::MeasureBase.Dirac, ::Analytic, rng) =
    _estimate(MeasureBase.atomof(prior))
function _evaluate(
    s::Probability{<:IntervalEvent},
    prior::MeasureBase.Dirac,
    ::Analytic,
    rng,
)
    _estimate(Int(s.event(MeasureBase.atomof(prior))))
end
function _evaluate(s::Probability{<:IntervalEvent}, prior, ::Analytic, rng)
    _estimate(MeasureBase.smf(prior, s.event.lower, s.event.upper))
end
_evaluate(s::Statistic, prior, ::Analytic, rng) = _estimate(s.f(prior))
function _evaluate(s::Probability, prior, plan::MonteCarlo, rng)
    rng === nothing &&
        throw(ArgumentError("Monte Carlo evaluation requires an explicit RNG"))
    hits = count(_ -> s.event(rand(rng, prior)), 1:plan.n)
    p = hits / plan.n
    error = sqrt(log(2 / plan.alpha) / (2 * plan.n))
    Estimate(
        p;
        lower=max(0, p-error),
        upper=min(1, p+error),
        evidence=:statistical,
        n=plan.n,
    )
end
