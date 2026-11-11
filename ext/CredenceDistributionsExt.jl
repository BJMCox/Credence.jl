module CredenceDistributionsExt

using Credence
using Distributions: DiscreteUnivariateDistribution, cdf, ccdf, quantile
using Distributions:
    Distributions,
    Normal,
    LogNormal,
    Beta,
    LocationScale,
    truncated,
    MvNormal,
    MvLogNormal,
    Gamma,
    Weibull,
    Exponential,
    TDist,
    Cauchy,
    Laplace,
    Dirichlet
using LinearAlgebra
using Statistics: Statistics, mean, cov
using MeasureBase: MeasureBase, AsMeasure, asmeasure
using Random: Random, AbstractRNG
using VarNames

Credence._covariance(prior::AsMeasure) = cov(prior.obj)
Credence._factor_template(prior::AsMeasure{<:Distributions.AbstractMvTDist}) =
    copy(Distributions.params(prior.obj)[2])
Credence._law_optic(prior::AsMeasure, ::VarNames.Iden) = prior
function Credence._law_optic(prior::AsMeasure, optic)
    law = prior.obj
    law isa Union{MvNormal,MvLogNormal,Dirichlet,Distributions.AbstractMvTDist} ||
        throw(Credence.NoAnalytic(optic))
    indices = VarNames.ohead(optic)(collect(1:length(law)))
    part = _indexed(law, indices)
    Credence._law_optic(asmeasure(part), VarNames.otail(optic))
end

function _indexed(law::Distributions.AbstractMvTDist, index::Integer)
    df, location, scatter = Distributions.params(law)
    LocationScale(location[index], sqrt(scatter[index, index]), TDist(df))
end
function _indexed(law::Distributions.AbstractMvTDist, indices)
    df, location, scatter = Distributions.params(law)
    Distributions.MvTDist(df, collect(location[indices]), Matrix(scatter[indices, indices]))
end
function _indexed(law::Dirichlet, index::Integer)
    alpha = first(Distributions.params(law))
    remainder = sum(alpha[j] for j in eachindex(alpha) if j != index)
    Beta(alpha[index], remainder)
end
struct DirichletMarginal{D,I} <: MeasureBase.AbstractMeasure
    parent::D
    indices::I
end
_indexed(law::Dirichlet, indices) = DirichletMarginal(law, collect(indices))
MeasureBase.massof(law::DirichletMarginal) = one(eltype(law.parent))
MeasureBase.mspace_elsize(law::DirichletMarginal) = size(law.indices)
Statistics.mean(law::DirichletMarginal) = mean(law.parent)[law.indices]
Statistics.var(law::DirichletMarginal) = Statistics.var(law.parent)[law.indices]
Statistics.std(law::DirichletMarginal) = sqrt.(Statistics.var(law))
Credence._covariance(law::DirichletMarginal) = cov(law.parent)[law.indices, law.indices]
Random.rand(rng::AbstractRNG, law::DirichletMarginal) =
    Random.rand(rng, law.parent)[law.indices]
Credence._law_optic(law::DirichletMarginal, ::VarNames.Iden) = law
function Credence._law_optic(law::DirichletMarginal, optic)
    indices = VarNames.ohead(optic)(law.indices)
    part = _indexed(law.parent, indices)
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

_scalar_spec(::Val{:gamma}) = (Gamma, (shape=2.0, scale=1.0))
_scalar_spec(::Val{:weibull}) = (Weibull, (shape=2.0, scale=1.0))
_scalar_spec(::Val{:exponential}) = (Exponential, (scale=1.0,))
_scalar_spec(::Val{:student_t}) = (
    (df, location, scale) -> LocationScale(location, scale, TDist(df)),
    (df=4.0, location=0.0, scale=1.0),
)
_scalar_spec(::Val{:cauchy}) = (Cauchy, (location=0.0, scale=1.0))
_scalar_spec(::Val{:laplace}) = (Laplace, (location=0.0, scale=1.0))
_scalar_spec(::Val{:half_normal}) = (
    scale ->
        truncated(Normal(zero(scale), scale), zero(scale), oftype(float(scale), Inf)),
    (scale=1.0,),
)
_scalar_spec(::Val{:half_student_t}) = (
    (df, scale) -> truncated(
        LocationScale(zero(scale), scale, TDist(df)),
        zero(scale),
        oftype(float(scale), Inf),
    ),
    (df=4.0, scale=1.0),
)
_coordinate(name) = name == :location ? name : Symbol(:log_, name)
function _initial_parameter(name, value)
    if name == :location
        return value
    end
    log(value)
end
function _parameter(parameters, fixed, ::Val{name}) where {name}
    value = getproperty(fixed, name)
    value === nothing || return value
    coordinate = getproperty(parameters, _coordinate(name))
    if name == :location
        return coordinate
    end
    exp(coordinate)
end
function Credence._family_candidate(id, family::Credence.ScalarFamily{K}) where {K}
    constructor, initial = _scalar_spec(Val(K))
    _scalar_candidate(id, family, constructor, family.parameters, (initial,))
end
function _scalar_candidate(id, family, constructor, fixed, initials)
    names = keys(first(initials))
    tags = map(Val, names)
    free = filter(name -> getproperty(fixed, name) === nothing, names)
    defaults = map(initials) do initial
        NamedTuple{map(_coordinate, free)}(
            map(name -> _initial_parameter(name, getproperty(initial, name)), free),
        )
    end
    if isempty(free)
        defaults = (first(defaults),)
    end
    build = p -> asmeasure(constructor(map(tag -> _parameter(p, fixed, tag), tags)...))
    Credence.Candidate(id, build, _starts(family.starts, defaults); family)
end

function Credence._family_candidate(id, family::Credence.DirichletFamily)
    n = family.dimension
    initial, alpha = if family.mean !== nothing
        if family.concentration === nothing
            ((log_concentration=log(float(n)),), p -> exp(p.log_concentration) .* family.mean)
        else
            (NamedTuple(), _ -> family.concentration .* family.mean)
        end
    elseif family.concentration !== nothing
        function concentration_parameters(p)
            logits = vcat(p.logits, zero(eltype(p.logits)))
            weights = exp.(logits .- maximum(logits))
            family.concentration .* (weights ./ sum(weights))
        end
        ((logits=zeros(n-1),), concentration_parameters)
    else
        ((log_alpha=zeros(n),), p -> exp.(p.log_alpha))
    end
    build = p -> asmeasure(Dirichlet(alpha(p)))
    Credence.Candidate(id, build, _starts(family.starts, (initial,)); family)
end

function Credence._family_candidate(id, family::Credence.MvStudentTFamily)
    n = family.dimension
    initial = (location=zeros(n), log_scales=zeros(n), lower=zeros(n*(n-1)÷2))
    if family.df === nothing
        initial = (; log_df=log(4.0), initial...)
    end
    function build(p)
        factor = _cholesky(p)
        df = family.df
        if df === nothing
            df = exp(p.log_df)
        end
        asmeasure(Distributions.MvTDist(df, collect(p.location), Matrix(factor * factor')))
    end
    Credence.Candidate(id, build, _starts(family.starts, (initial,)); family)
end
for (family, law) in
    ((Credence.NormalFamily, Normal), (Credence.LogNormalFamily, LogNormal))
    @eval function Credence._family_candidate(id, family::$family)
        initial = ((location=0.0, scale=1.0), (location=0.5, scale=exp(-0.5)))
        fixed = (location=family.location, scale=family.scale)
        _scalar_candidate(id, family, $law, fixed, initial)
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
    Credence.Candidate(id, build, _starts(family.starts, defaults); family)
end
_adapter(family::Credence.AbstractFamily) = family
_adapter(::Type{Normal}) = Credence.NormalFamily()
_adapter(::Type{LogNormal}) = Credence.LogNormalFamily()
_adapter(::Type{Gamma}) = Credence.GammaFamily()
_adapter(::Type{Weibull}) = Credence.WeibullFamily()
_adapter(::Type{Exponential}) = Credence.ExponentialFamily()
_adapter(::Type{TDist}) = Credence.StudentTFamily()
_adapter(::Type{Cauchy}) = Credence.CauchyFamily()
_adapter(::Type{Laplace}) = Credence.LaplaceFamily()
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
        initial = (location=center,)
        if base.scale === nothing
            initial = (; initial..., log_scale=log(spread))
        end
        starts = (initial,)
    end
    build = p -> asmeasure(truncated(candidate.build(p).obj, family.lower, family.upper))
    Credence.Candidate(id, build, starts; family)
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
        Credence.Candidate(id, build, _starts(family.starts, defaults); family)
    end
end

function Credence._evaluate(
    statistic::Credence.Quantile,
    prior::AsMeasure{<:Union{MvNormal,MvLogNormal,Dirichlet,Distributions.AbstractMvTDist}},
    ::Credence.Analytic,
    rng,
)
    Credence._estimate([
        quantile(_indexed(prior.obj, i), statistic.p) for i in 1:length(prior.obj)
    ])
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

Credence._flatppl(prior::AsMeasure, path) = Credence._flatppl(prior.obj, path)
for (law, name) in (
    (Normal, "Normal"),
    (LogNormal, "LogNormal"),
    (Beta, "Beta"),
    (MvNormal, "MvNormal"),
    (Weibull, "Weibull"),
    (Cauchy, "Cauchy"),
    (Laplace, "Laplace"),
    (Dirichlet, "Dirichlet"),
    (TDist, "StudentT"),
)
    @eval function Credence._flatppl(prior::$law, path)
        arguments = map(Distributions.params(prior)) do parameter
            Credence._flatppl_value(parameter, path)
        end
        $name * "(" * join(arguments, ", ") * ")"
    end
end
function _flatppl_rate(scale, path)
    rate = inv(scale)
    isfinite(rate) && rate > 0 || throw(
        ArgumentError("scale at $path has no representable positive finite FlatPPL rate"),
    )
    Credence._flatppl_value(rate, path)
end
function Credence._flatppl(prior::Gamma, path)
    shape, scale = Distributions.params(prior)
    "Gamma(" *
    Credence._flatppl_value(shape, path) *
    ", " *
    _flatppl_rate(scale, path) *
    ")"
end
function Credence._flatppl(prior::Exponential, path)
    scale = only(Distributions.params(prior))
    "Exponential(" * _flatppl_rate(scale, path) * ")"
end
function Credence._flatppl(prior::MvLogNormal, path)
    location, covariance = Distributions.params(prior)
    "pushfwd(fn(exp.(_)), " * Credence._flatppl(MvNormal(location, covariance), path) * ")"
end
function Credence._flatppl(prior::LocationScale, path)
    location, scale, base = Distributions.params(prior)
    "locscale(" *
    Credence._flatppl(base, path) *
    ", " *
    Credence._flatppl_value(location, path) *
    ", " *
    Credence._flatppl_value(scale, path) *
    ")"
end
function Credence._flatppl(prior::Distributions.Truncated, path)
    "normalize(truncate(" *
    Credence._flatppl(prior.untruncated, path) *
    ", interval(" *
    Credence._flatppl_value(prior.lower, path) *
    ", " *
    Credence._flatppl_value(prior.upper, path) *
    ")))"
end

end
