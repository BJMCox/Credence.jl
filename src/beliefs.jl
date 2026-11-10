abstract type Relation end

"""
    AtLeast(value)

Require a statistic to be at least the stated value.
"""
struct AtLeast{T} <: Relation
    value::T
end
"""
    AtMost(value)

Require a statistic to be at most the stated value.
"""
struct AtMost{T} <: Relation
    value::T
end
"""
    Between(lower, upper)

Require a statistic to lie in the closed interval [lower, upper].
"""
struct Between{L,U} <: Relation
    lower::L
    upper::U
    function Between(lower, upper)
        lower <= upper || throw(ArgumentError("bounds must increase"))
        new{typeof(lower),typeof(upper)}(lower, upper)
    end
end
"""
    EqualTo(value)

Require exact equality. Use `Between` to express an expert's tolerance.
"""
struct EqualTo{T} <: Relation
    value::T
end
_bounds(r::AtLeast) = (r.value, Inf)
_bounds(r::AtMost) = (-Inf, r.value)
_bounds(r::Between) = (r.lower, r.upper)
_bounds(r::EqualTo) = (r.value, r.value)

"""
    Hard(id, statistic, relation)

A named hard requirement on a statistic.
"""
struct Hard{I,S<:AbstractStatistic,R<:Relation}
    id::I
    statistic::S
    relation::R
end

"""
    Approximate(id, statistic, target; scale, importance=nothing)

An approximate point target with an explicit scale and optional importance weight.
"""
struct Approximate{I,S<:AbstractStatistic,T,C,W}
    id::I
    statistic::S
    target::T
    scale::C
    importance::W
    function Approximate(
        id,
        statistic::AbstractStatistic,
        target;
        scale,
        importance=nothing,
    )
        isfinite(target) || throw(ArgumentError("target must be finite"))
        isfinite(scale) && scale > 0 ||
            throw(ArgumentError("scale must be positive and finite"))
        if importance !== nothing
            isfinite(importance) && importance >= 0 ||
                throw(ArgumentError("importance must be nonnegative and finite"))
        end
        new{typeof(id),typeof(statistic),typeof(target),typeof(scale),typeof(importance)}(
            id,
            statistic,
            target,
            scale,
            importance,
        )
    end
end
