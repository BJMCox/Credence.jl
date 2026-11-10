module CredenceDistributionsExt

using Credence
using Distributions: DiscreteUnivariateDistribution, cdf, ccdf, quantile
using MeasureBase: AsMeasure

function Credence._evaluate(
    statistic::Credence.Quantile,
    prior::AsMeasure{<:DiscreteUnivariateDistribution},
    ::Credence.Analytic,
    rng,
)
    return Credence._estimate(quantile(prior.obj, statistic.p))
end

function Credence._evaluate(
    statistic::Credence.Probability{<:Credence.IntervalEvent},
    prior::AsMeasure{<:DiscreteUnivariateDistribution},
    ::Credence.Analytic,
    rng,
)
    lower, upper = statistic.event.lower, statistic.event.upper
    lower_cdf = cdf(prior.obj, lower)
    probability = if lower_cdf <= 0.5
        cdf(prior.obj, upper) - lower_cdf
    else
        ccdf(prior.obj, lower) - ccdf(prior.obj, upper)
    end
    return Credence._estimate(probability)
end

end
