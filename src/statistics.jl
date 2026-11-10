abstract type AbstractStatistic end

"An evaluated statistic, its enclosure, and the source of its evidence."
struct Estimate{V,L,U}
    value::V
    lower::L
    upper::U
    evidence::Symbol
    function Estimate(value; lower=value, upper=value, evidence=:analytic)
        lower <= value <= upper ||
            throw(ArgumentError("estimate lies outside its enclosure"))
        new{typeof(value),typeof(lower),typeof(upper)}(value, lower, upper, evidence)
    end
end

struct IntervalEvent{A,B}
    lower::A
    upper::B
end
(event::IntervalEvent)(x) = event.lower < x <= event.upper

"`Probability(a, b)` denotes P(a < X <= b). A predicate requires a sampling evaluation plan."
struct Probability{F} <: AbstractStatistic
    event::F
end
function Probability(lower::Real, upper::Real)
    lower < upper || throw(ArgumentError("interval endpoints must increase"))
    Probability(IntervalEvent(lower, upper))
end

"The lower quantile at probability p, with 0 < p < 1."
struct Quantile{T} <: AbstractStatistic
    p::T
    function Quantile(p::Real)
        0 < p < 1 || throw(ArgumentError("quantile probability must lie in (0,1)"))
        new{typeof(p)}(p)
    end
end

"The population mean, when it exists."
struct Mean <: AbstractStatistic end
"The population standard deviation, when it exists."
struct Std <: AbstractStatistic end

"A user-supplied analytic statistic `f(measure)`, optionally returning an Estimate."
struct Statistic{F,L} <: AbstractStatistic
    f::F
    label::L
end
Statistic(f; label=:custom) = Statistic(f, label)

"Evaluate through analytic measure operations, without sampling."
struct Analytic end

"Validate event probabilities from n independent draws with failure probability alpha."
struct MonteCarlo{T}
    n::Int
    alpha::T
    function MonteCarlo(n::Integer; alpha=0.01)
        n > 0 || throw(ArgumentError("sample count must be positive"))
        0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0,1)"))
        new{typeof(alpha)}(n, alpha)
    end
end

"Evaluate a statistic under a normalized MeasureBase measure."
function evaluate(statistic::AbstractStatistic, prior; plan=Analytic(), rng=nothing)
    isone(MeasureBase.massof(prior)) || throw(ArgumentError("candidate must be normalized"))
    _evaluate(statistic, prior, plan, rng)
end

function _estimate(x::Estimate)
    isfinite(x.value) || throw(ArgumentError("statistic is not finite"))
    x
end
function _estimate(x::Real)
    isfinite(x) || throw(ArgumentError("statistic is not finite"))
    Estimate(x)
end
_evaluate(s::Mean, prior, ::Analytic, rng) = _estimate(mean(prior))
_evaluate(s::Std, prior, ::Analytic, rng) = _estimate(std(prior))
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
    Estimate(p; lower=max(0, p-error), upper=min(1, p+error), evidence=:statistical)
end
