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
        _check_bounds(lower, upper)
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
    function Hard(id, statistic::AbstractStatistic, relation::Relation)
        _check_symmetric(statistic, relation)
        _check_bounds(_bounds(relation)..., statistic)
        new{typeof(id),typeof(statistic),typeof(relation)}(id, statistic, relation)
    end
end

"""
    Approximate(id, statistic, target; scale, importance=1)

An approximate point target with an explicit scale and optional importance weight.
"""
struct Approximate{I,S<:AbstractStatistic,T,C,W,E,D}
    id::I
    statistic::S
    target::T
    scale::C
    importance::W
    entries::E
    diagnostics::D
    function Approximate(id, statistic::AbstractStatistic, target; scale, importance=1)
        _check_symmetric(statistic, target; point=true)
        _check_symmetric(statistic, scale)
        _check_shape(target, scale)
        entries = _entries(statistic, target)
        isempty(entries) &&
            throw(ArgumentError("a belief needs at least one supplied free entry"))
        for index in entries
            isfinite(_at(statistic, target, index)) ||
                throw(ArgumentError("target must be finite"))
            value = _at(statistic, scale, index)
            !_absent(value) && isfinite(value) && value > 0 ||
                throw(ArgumentError("every supplied target needs a positive finite scale"))
        end
        if importance !== nothing
            isfinite(importance) && importance >= 0 ||
                throw(ArgumentError("importance must be nonnegative and finite"))
        end
        diagnostics = _matrix_diagnostics(statistic, target)
        new{
            typeof(id),
            typeof(statistic),
            typeof(target),
            typeof(scale),
            typeof(importance),
            typeof(entries),
            typeof(diagnostics),
        }(
            id,
            statistic,
            target,
            scale,
            importance,
            entries,
            diagnostics,
        )
    end
end

_absent(x) = x === nothing || ismissing(x)
_base_statistic(s) = s
_symmetric(s) = _base_statistic(s) isa Union{CovarianceMatrix,CorrelationMatrix}
_correlation(s) = _base_statistic(s) isa CorrelationMatrix
_at(x, ::Nothing) = x
_at(x::Number, ::Nothing) = x
_at(::Nothing, ::Nothing) = nothing
_at(::Missing, ::Nothing) = missing
_at(x::Number, index) = x
_at(::Nothing, index) = nothing
_at(::Missing, index) = missing
_at(x::AbstractArray, index) = x[index]
_at(statistic, x, index) = _at(x, index)
function _at(statistic, x::Union{UpperTriangular,LowerTriangular}, index::CartesianIndex{2})
    _symmetric(statistic) || return x[index]
    i, j = Tuple(index)
    x isa UpperTriangular ? x[min(i, j), max(i, j)] : x[max(i, j), min(i, j)]
end
_indices(x::AbstractArray) = CartesianIndices(x)
_indices(x) = (nothing,)
_same(a, b) = (_absent(a) && _absent(b)) || (!_absent(a) && !_absent(b) && a == b)

function _check_shape(a, b)
    if a isa AbstractArray && b isa AbstractArray
        axes(a) == axes(b) ||
            throw(DimensionMismatch("belief arrays must have matching axes"))
    end
    nothing
end

function _check_bounds(lower, upper, statistic=nothing)
    _check_shape(lower, upper)
    if statistic === nothing && (
        lower isa Union{UpperTriangular,LowerTriangular} ||
        upper isa Union{UpperTriangular,LowerTriangular}
    )
        return nothing
    end
    template = lower isa AbstractArray ? lower : upper
    for index in _indices(template)
        lo, hi = _at(statistic, lower, index), _at(statistic, upper, index)
        _absent(lo) ||
            _absent(hi) ||
            lo <= hi ||
            throw(ArgumentError("bounds must increase"))
    end
end

function _check_symmetric(statistic, x; point=false)
    _symmetric(statistic) && x isa AbstractArray || return nothing
    x isa AbstractMatrix && size(x, 1) == size(x, 2) ||
        throw(DimensionMismatch("matrix statistics require square targets"))
    if !(x isa Union{UpperTriangular,LowerTriangular})
        for j in axes(x, 2), i in axes(x, 1)
            _same(x[i, j], x[j, i]) ||
                throw(ArgumentError("mirrored matrix entries must agree"))
        end
    end
    if point && _correlation(statistic)
        all(i -> _absent(x[i, i]) || x[i, i] == 1, axes(x, 1)) ||
            throw(ArgumentError("correlation target diagonals must equal one"))
    end
    nothing
end
function _check_symmetric(statistic, relation::Relation; point=false)
    lower, upper = _bounds(relation)
    _check_symmetric(statistic, lower)
    _check_symmetric(statistic, upper)
    template = lower isa AbstractArray ? lower : upper
    if _correlation(statistic) && template isa AbstractMatrix
        for i in axes(template, 1)
            lo, hi = _at(statistic, lower, CartesianIndex(i, i)),
            _at(statistic, upper, CartesianIndex(i, i))
            (_absent(lo) || lo <= 1) && (_absent(hi) || 1 <= hi) ||
                throw(ArgumentError("correlation diagonal bounds must admit one"))
        end
    end
    nothing
end

function _entries(statistic, target)
    filter(_indices(target)) do index
        supplied = !_absent(_at(statistic, target, index))
        if _symmetric(statistic) && index isa CartesianIndex{2}
            i, j = Tuple(index)
            return supplied && i <= j && (!_correlation(statistic) || i != j)
        end
        supplied
    end
end

function _matrix_diagnostics(statistic, target)
    _symmetric(statistic) && target isa AbstractMatrix || return ()
    any(i -> _absent(_at(statistic, target, i)), _indices(target)) && return ()
    matrix = map(i -> _at(statistic, target, i), CartesianIndices(target))
    smallest = try
        eigmin(Symmetric(matrix))
    catch error
        error isa MethodError || rethrow()
        return (:matrix_definiteness_unchecked,)
    end
    smallest < 0 ? (:inconsistent_matrix_target,) : ()
end

function _discrepancy(belief::Approximate, value)
    (belief.target isa AbstractArray) == (value isa AbstractArray) ||
        throw(DimensionMismatch("statistic and target must have the same shape"))
    _check_shape(belief.target, value)
    total = sum(belief.entries) do index
        abs2(
            (
                _at(belief.statistic, value, index) -
                _at(belief.statistic, belief.target, index)
            ) / _at(belief.statistic, belief.scale, index),
        )
    end
    total / length(belief.entries)
end

function _hard_entries(belief::Hard, value)
    lower, upper = _bounds(belief.relation)
    if !(value isa AbstractArray) && (lower isa AbstractArray || upper isa AbstractArray)
        throw(DimensionMismatch("array bounds require an array-valued statistic"))
    end
    _check_shape(value, lower)
    _check_shape(value, upper)
    filter(_entries(belief.statistic, value)) do index
        !_absent(_at(belief.statistic, lower, index)) ||
            !_absent(_at(belief.statistic, upper, index))
    end
end
function _entry_bounds(belief, index)
    lower, upper = _bounds(belief.relation)
    lo, hi = _at(belief.statistic, lower, index), _at(belief.statistic, upper, index)
    (_absent(lo) ? -Inf : lo, _absent(hi) ? Inf : hi)
end
