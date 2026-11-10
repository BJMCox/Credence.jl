abstract type AbstractFamily end

"""A Normal family with fitted location and positive scale."""
struct NormalFamily{M,S} <: AbstractFamily
    location::M
    starts::S
end
NormalFamily(; location=nothing, starts=nothing) = NormalFamily(location, starts)
"""A LogNormal family with fitted log location and positive log scale."""
struct LogNormalFamily{M,S} <: AbstractFamily
    location::M
    starts::S
end
LogNormalFamily(; location=nothing, starts=nothing) = LogNormalFamily(location, starts)

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
    Candidate(id, build, candidate.starts)
end

function _grouped_candidates(groups)
    Tuple(
        Iterators.flatten(
            map(groups) do group
                target = _target(group.first)
                target isa Derived &&
                    throw(ArgumentError("derived quantities do not define prior families"))
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
