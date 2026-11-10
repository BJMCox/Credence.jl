"""`Independent(reference => family_id, ...)` declares one independent combination.

Use `reference => selection` to keep an accepted marginal law fixed.
"""
struct Independent{C}
    choices::C
end
Independent(choices::Pair...) = Independent(choices)

"""Explicitly enumerate all marginal family combinations with independent dependence."""
struct AllCombinations end

struct IndependentBuilder{P}
    parts::P
end
_factor_scope(candidate::Candidate) = candidate.scope
_factor_law(candidate::Candidate, parameters) = candidate.build(parameters)
_factor_length(candidate::Candidate) = length(_flatten(first(candidate.starts)))

function _independent_law(parts)
    refs = first.(parts)
    any(r -> r isa Joint, refs) && return _record_product(parts)
    all(r -> r isa VarName, refs) ||
        throw(ArgumentError("independent factors need variable references"))
    if all(r -> getoptic(r) isa VarNames.Iden, refs)
        names = getsym.(refs)
        return productmeasure(NamedTuple{names}(last.(parts)))
    end
    record = VarNames.VarNamedTuple(parts)
    names = Tuple(unique(getsym.(refs)))
    laws = map(names) do name
        value = record[VarName{name}()]
        _structured_law(value)
    end
    productmeasure(NamedTuple{names}(laws))
end
_structured_law(value::MeasureBase.AbstractMeasure) = value
_structured_law(value::AbstractArray) = productmeasure(value)
function _structured_law(value::VarNames.VarNamedTuple)
    refs = Tuple(keys(value))
    _independent_law(map(ref -> ref=>value[ref], refs))
end

function (builder::IndependentBuilder)(parameters)
    offset = 0
    factors = map(builder.parts) do part
        n = _factor_length(part)
        values = view(parameters, (offset+1):(offset+n))
        offset += n
        _factor_scope(part) => _factor_law(part, values)
    end
    _independent_law(factors)
end

function _independent_candidate(candidates, id, composition::Independent)
    parts = map(composition.choices) do choice
        target = _target(choice.first)
        if choice.second isa Selection
            selection = choice.second
            selection.role == :accepted ||
                throw(ArgumentError("locked marginals require accepted selections"))
            _factor_scope(selection) == target ||
                throw(ArgumentError("selection target differs from its binding"))
            return selection
        elseif choice.second isa Candidate
            choice.second.scope == target ||
                throw(ArgumentError("candidate target differs from its binding"))
            return choice.second
        end
        matches = filter(c -> c.scope == target && c.id == choice.second, candidates)
        only(matches)
    end
    refs = Tuple(Iterators.flatten(map(p -> _references(_factor_scope(p)), parts)))
    for i in eachindex(refs), j in (i+1):length(refs)
        (subsumes(refs[i], refs[j]) || subsumes(refs[j], refs[i])) &&
            throw(ArgumentError("independent factor bindings overlap"))
    end
    count = maximum(p -> p isa Selection ? 1 : length(p.starts), parts)
    points = ntuple(count) do i
        starts = map(parts) do part
            part isa Selection && return ()
            part.starts[mod1(i, length(part.starts))]
        end
        chunks = filter(x -> !isempty(x), map(_flatten, starts))
        isempty(chunks) ? Float64[] : reduce(vcat, chunks)
    end
    lower = _combined_bounds(parts, :lower)
    upper = _combined_bounds(parts, :upper)
    candidate = Candidate(id, IndependentBuilder(parts), points; lower, upper)
    _scoped(candidate, refs)
end

_requirements(candidate::Candidate) = candidate.requirements
function _requirements(candidate::Candidate{I,<:IndependentBuilder}) where {I}
    (
        candidate.requirements...,
        Tuple(Iterators.flatten(map(_requirements, candidate.build.parts)))...,
    )
end
function _combined_bounds(parts, field)
    any(p -> p isa Candidate && getproperty(p, field) !== nothing, parts) || return nothing
    pieces = map(parts) do part
        part isa Selection && return Float64[]
        bound = getproperty(part, field)
        bound === nothing &&
            return fill(field == :lower ? -Inf : Inf, _factor_length(part))
        collect(bound)
    end
    reduce(vcat, pieces)
end
function _inherit_requirements(beliefs, candidates)
    result = beliefs
    for candidate in candidates, belief in _requirements(candidate)
        index = findfirst(b -> b.id == belief.id, result)
        if index === nothing
            result = (result..., belief)
        else
            old = result[index]
            old isa Hard &&
            isequal(old.statistic, belief.statistic) &&
            isequal(_bounds(old.relation), _bounds(belief.relation)) || throw(
                ArgumentError("a retained hard belief conflicts with its new definition"),
            )
        end
    end
    result
end
function _combinations(candidates, combinations)
    map(Tuple(combinations)) do choice
        choice isa Candidate && return choice
        _independent_candidate(candidates, choice.first, choice.second)
    end
end
function _combinations(candidates, ::AllCombinations)
    targets = Tuple(unique(map(c -> c.scope, candidates)))
    groups = map(t -> filter(c -> c.scope == t, candidates), targets)
    map(Tuple(Iterators.product(groups...))) do choices
        names = map(c -> c.id, choices)
        _independent_candidate(
            candidates,
            names,
            Independent(map(c -> c.scope=>c.id, choices)...),
        )
    end
end
