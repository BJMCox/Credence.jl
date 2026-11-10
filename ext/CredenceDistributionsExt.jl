module CredenceDistributionsExt

using Credence
using Distributions: DiscreteUnivariateDistribution, cdf, ccdf, quantile
using Distributions:
    Distributions, Normal, LogNormal, Beta, LocationScale, truncated, MvNormal, MvLogNormal
using LinearAlgebra
using Statistics: mean, cov
using MeasureBase: AsMeasure, asmeasure
using VarNames

Credence._covariance(prior::AsMeasure) = cov(prior.obj)
Credence._law_optic(prior::AsMeasure, ::VarNames.Iden) = prior
function Credence._law_optic(prior::AsMeasure, optic)
    law = prior.obj
    law isa Union{MvNormal,MvLogNormal} || throw(Credence.NoAnalytic(optic))
    indices = VarNames.ohead(optic)(collect(1:length(law)))
    part = _indexed(law, indices)
    Credence._law_optic(asmeasure(part), VarNames.otail(optic))
end
function _indexed(law::MvNormal, index::Integer)
    Normal(mean(law)[index], sqrt(cov(law)[index, index]))
end
_indexed(law::MvNormal, indices) = MvNormal(mean(law)[indices], cov(law)[indices, indices])
function _indexed(law::MvLogNormal, index::Integer)
    location, scale = Distributions.params(law)
    LogNormal(location[index], sqrt(scale[index, index]))
end
function _indexed(law::MvLogNormal, indices)
    location, scale = Distributions.params(law)
    MvLogNormal(location[indices], scale[indices, indices])
end

_starts(starts, default) =
    starts === nothing ? default : (starts isa NamedTuple ? (starts,) : starts)
for (family, law) in
    ((Credence.NormalFamily, Normal), (Credence.LogNormalFamily, LogNormal))
    @eval function Credence._family_candidate(id, family::$family)
        default = if family.location === nothing
            ((location=0.0, log_scale=0.0), (location=0.5, log_scale=-0.5))
        else
            ((log_scale=0.0,), (log_scale=-0.5,))
        end
        build =
            p -> asmeasure(
                $law(
                    family.location === nothing ? p.location : family.location,
                    exp(p.log_scale),
                ),
            )
        Credence.Candidate(id, build, _starts(family.starts, default))
    end
end
function Credence._family_candidate(id, family::Credence.ScaledBetaFamily)
    build =
        p -> asmeasure(
            LocationScale(
                family.lower,
                family.upper-family.lower,
                Beta(exp(p.log_alpha), exp(p.log_beta)),
            ),
        )
    defaults = ((log_alpha=0.5, log_beta=1.0), (log_alpha=1.0, log_beta=1.5))
    Credence.Candidate(id, build, _starts(family.starts, defaults))
end
_adapter(family::Credence.AbstractFamily) = family
_adapter(::Type{Normal}) = Credence.NormalFamily()
_adapter(::Type{LogNormal}) = Credence.LogNormalFamily()
function Credence._family_candidate(id, family::Credence.TruncatedFamily)
    base = _adapter(family.base)
    candidate = Credence._family_candidate(id, base)
    starts = candidate.starts
    if family.starts !== nothing
        starts = _starts(family.starts, starts)
    elseif base isa Union{Credence.NormalFamily,Credence.LogNormalFamily} &&
           base.location === nothing &&
           isfinite(family.lower) &&
           isfinite(family.upper)
        center = (family.lower+family.upper)/2
        if base isa Credence.LogNormalFamily
            center = log(max(center, eps(float(family.upper))))
        end
        spread = base isa Credence.LogNormalFamily ? 0.7 : (family.upper-family.lower)/4
        starts = ((location=center, log_scale=log(spread)),)
    end
    build = p -> asmeasure(truncated(candidate.build(p).obj, family.lower, family.upper))
    Credence.Candidate(id, build, starts)
end

function _cholesky(p)
    n = length(p.log_scales)
    factor = diagm(exp.(p.log_scales))
    k = 0
    for j in 1:n, i in (j+1):n
        k += 1
        factor[i, j] = p.lower[k]
    end
    factor
end
for (family, law) in
    ((Credence.MvNormalFamily, MvNormal), (Credence.MvLogNormalFamily, MvLogNormal))
    @eval function Credence._family_candidate(id, family::$family)
        n = family.dimension
        n > 0 || throw(ArgumentError("multivariate dimension must be positive"))
        defaults = ((location=zeros(n), log_scales=zeros(n), lower=zeros(n*(n-1)÷2)),)
        build = p -> begin
            factor = _cholesky(p)
            asmeasure($law(p.location, Symmetric(factor*factor')))
        end
        Credence.Candidate(id, build, _starts(family.starts, defaults))
    end
end

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
