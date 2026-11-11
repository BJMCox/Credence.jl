abstract type AbstractFamily end

"""`NormalFamily(; location=nothing, scale=nothing, starts=nothing)` fits a Normal law with optional fixed location or scale."""
struct NormalFamily{M,C,S} <: AbstractFamily
    location::M
    scale::C
    starts::S
end
function NormalFamily(; location=nothing, scale=nothing, starts=nothing)
    _check_fixed_scale(scale)
    NormalFamily(location, scale, starts)
end
NormalFamily(location, starts) = NormalFamily(location, nothing, starts)
"""`LogNormalFamily(; location=nothing, scale=nothing, starts=nothing)` fits a LogNormal law with optional fixed log location or log standard deviation."""
struct LogNormalFamily{M,C,S} <: AbstractFamily
    location::M
    scale::C
    starts::S
end
function LogNormalFamily(; location=nothing, scale=nothing, starts=nothing)
    _check_fixed_scale(scale)
    LogNormalFamily(location, scale, starts)
end
LogNormalFamily(location, starts) = LogNormalFamily(location, nothing, starts)
function _check_fixed_scale(scale)
    scale === nothing ||
        (isfinite(scale) && scale > 0) ||
        throw(ArgumentError("fixed scale must be positive and finite"))
end

struct ScalarFamily{K,P,S} <: AbstractFamily
    parameters::P
    starts::S
end
function _scalar_family(kind::Symbol, parameters, starts)
    for (name, value) in pairs(parameters)
        value === nothing && continue
        isfinite(value) && (name == :location || value > 0) ||
            throw(ArgumentError("fixed $name must be finite and in its parameter domain"))
    end
    ScalarFamily{kind,typeof(parameters),typeof(starts)}(parameters, starts)
end

"""`GammaFamily(; shape=nothing, scale=nothing, starts=nothing)` fits a shape-scale Gamma prior."""
GammaFamily(; shape=nothing, scale=nothing, starts=nothing) =
    _scalar_family(:gamma, (; shape, scale), starts)
"""`WeibullFamily(; shape=nothing, scale=nothing, starts=nothing)` fits a Weibull prior."""
WeibullFamily(; shape=nothing, scale=nothing, starts=nothing) =
    _scalar_family(:weibull, (; shape, scale), starts)
"""`ExponentialFamily(; scale=nothing, starts=nothing)` fits an Exponential prior with mean `scale`."""
ExponentialFamily(; scale=nothing, starts=nothing) =
    _scalar_family(:exponential, (; scale), starts)
"""`StudentTFamily(; df=nothing, location=nothing, scale=nothing, starts=nothing)` fits a location-scale Student-t prior."""
StudentTFamily(; df=nothing, location=nothing, scale=nothing, starts=nothing) =
    _scalar_family(:student_t, (; df, location, scale), starts)
StudentTFamily(df::Real; kwargs...) = StudentTFamily(; df, kwargs...)
"""`CauchyFamily(; location=nothing, scale=nothing, starts=nothing)` fits a Cauchy prior."""
CauchyFamily(; location=nothing, scale=nothing, starts=nothing) =
    _scalar_family(:cauchy, (; location, scale), starts)
"""`LaplaceFamily(; location=nothing, scale=nothing, starts=nothing)` fits a Laplace prior."""
LaplaceFamily(; location=nothing, scale=nothing, starts=nothing) =
    _scalar_family(:laplace, (; location, scale), starts)
"""`HalfNormalFamily(; scale=nothing, starts=nothing)` fits a Normal prior truncated to nonnegative values."""
HalfNormalFamily(; scale=nothing, starts=nothing) =
    _scalar_family(:half_normal, (; scale), starts)
"""`HalfStudentTFamily(; df=nothing, scale=nothing, starts=nothing)` fits a Student-t prior truncated to nonnegative values."""
HalfStudentTFamily(; df=nothing, scale=nothing, starts=nothing) =
    _scalar_family(:half_student_t, (; df, scale), starts)
HalfStudentTFamily(df::Real; kwargs...) = HalfStudentTFamily(; df, kwargs...)

"""
    DirichletFamily(dimension; mean=nothing, concentration=nothing, starts=nothing)

Fit a Dirichlet prior. Fix its mean vector or total concentration with the corresponding keyword.
"""
struct DirichletFamily{M,C,S} <: AbstractFamily
    dimension::Int
    mean::M
    concentration::C
    starts::S
end
function DirichletFamily(n; mean=nothing, concentration=nothing, starts=nothing)
    n >= 2 || throw(ArgumentError("a simplex needs at least two components"))
    if mean !== nothing
        length(mean) == n &&
        all(x -> isfinite(x) && x > 0, mean) &&
        isapprox(sum(mean), one(sum(mean))) ||
            throw(ArgumentError("fixed Dirichlet mean must be a positive simplex vector"))
    end
    concentration === nothing ||
        (isfinite(concentration) && concentration > 0) ||
        throw(ArgumentError("fixed concentration must be positive and finite"))
    DirichletFamily(n, mean, concentration, starts)
end

"""`MvStudentTFamily(dimension; df=nothing, starts=nothing)` fits location, scatter, and optionally degrees of freedom."""
struct MvStudentTFamily{D,S} <: AbstractFamily
    dimension::Int
    df::D
    starts::S
end
function MvStudentTFamily(n; df=nothing, starts=nothing)
    n > 0 || throw(ArgumentError("multivariate dimension must be positive"))
    df === nothing ||
        (isfinite(df) && df > 0) ||
        throw(ArgumentError("fixed degrees of freedom must be positive and finite"))
    MvStudentTFamily(n, df, starts)
end

"""`ScaledBetaFamily(lower, upper; starts=nothing)` fits a Beta law on this interval."""
struct ScaledBetaFamily{L,U,S} <: AbstractFamily
    lower::L
    upper::U
    starts::S
    function ScaledBetaFamily(lower, upper; starts=nothing)
        lower < upper || throw(ArgumentError("family support bounds must increase"))
        new{typeof(lower),typeof(upper),typeof(starts)}(lower, upper, starts)
    end
end

"""`TruncatedFamily(base, lower, upper; starts=nothing)` truncates a fitted family."""
struct TruncatedFamily{F,L,U,S} <: AbstractFamily
    base::F
    lower::L
    upper::U
    starts::S
    function TruncatedFamily(base, lower, upper; starts=nothing)
        lower < upper || throw(ArgumentError("truncation bounds must increase"))
        new{typeof(base),typeof(lower),typeof(upper),typeof(starts)}(
            base,
            lower,
            upper,
            starts,
        )
    end
end

"""`IIDFamily(family, dimensions)` repeats one fitted law with independent entries."""
struct IIDFamily{F,D} <: AbstractFamily
    family::F
    dimensions::D
end
IIDFamily(family, n::Integer) = IIDFamily(family, (n,))

"""A multivariate Normal with fitted mean and full positive definite covariance."""
struct MvNormalFamily{S} <: AbstractFamily
    dimension::Int
    starts::S
end
MvNormalFamily(n; starts=nothing) = MvNormalFamily(n, starts)
"""A multivariate LogNormal with fitted log mean and full log covariance."""
struct MvLogNormalFamily{S} <: AbstractFamily
    dimension::Int
    starts::S
end
MvLogNormalFamily(n; starts=nothing) = MvLogNormalFamily(n, starts)

function _family_candidate(id, family)
    throw(ArgumentError("load Distributions to use standard family adapters"))
end
function _family_candidate(id, family::IIDFamily)
    all(>(0), family.dimensions) || throw(ArgumentError("IID dimensions must be positive"))
    candidate = _family_candidate(id, family.family)
    build = p -> productmeasure(fill(candidate.build(p), family.dimensions))
    Candidate(id, build, candidate.starts; family)
end

function _grouped_candidates(groups)
    Tuple(
        Iterators.flatten(
            map(groups) do group
                target = _target(group.first)
                target isa Union{Derived,Predictive} && throw(
                    ArgumentError(
                        "derived and predictive quantities do not define prior families",
                    ),
                )
                map(Tuple(group.second)) do entry
                    candidate =
                        entry isa Pair ? _family_candidate(entry.first, entry.second) :
                        entry
                    _scoped(candidate, target)
                end
            end,
        ),
    )
end
