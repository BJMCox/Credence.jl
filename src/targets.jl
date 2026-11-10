_reference(x::Symbol) = VarName{x}()
_reference(x::VarName) = x

"""`Joint(references...)` selects ordered quantities for joint beliefs."""
struct Joint{R}
    references::R
    function Joint(references...)
        refs = map(_reference, references)
        isempty(refs) && throw(ArgumentError("Joint needs at least one reference"))
        length(unique(refs)) == length(refs) ||
            throw(ArgumentError("Joint references must be unique"))
        new{typeof(refs)}(refs)
    end
end
Base.isequal(a::Joint, b::Joint) = isequal(a.references, b.references)
Base.:(==)(a::Joint, b::Joint) = a.references == b.references
Base.hash(x::Joint, h::UInt) = hash(x.references, hash(Joint, h))

"""`Derived(name, references, f)` calls `f` with the selected values in order."""
struct Derived{I,R,F}
    name::I
    references::R
    f::F
end
function Derived(name, references::Union{Symbol,VarName,Tuple}, f)
    refs = references isa Tuple ? references : (references,)
    normalized = map(_reference, refs)
    Derived{typeof(name),typeof(normalized),typeof(f)}(name, normalized, f)
end

_target(x::Union{Symbol,VarName}) = _reference(x)
_target(x::Union{Joint,Derived}) = x
_targetid(x) = x
_targetid(x::Derived) = (:derived, x.name)
_references(x::VarName) = (x,)
_references(x::Union{Joint,Derived}) = x.references

struct On{T,S} <: AbstractStatistic
    target::T
    statistic::S
end
function On(target::Union{VarName,Joint,Derived}, statistic::AbstractStatistic)
    if statistic isa Union{Covariance,Correlation}
        target isa Joint && length(target.references) == 2 || throw(
            DimensionMismatch("pairwise statistics require Joint of two scalar quantities"),
        )
    end
    On{typeof(target),typeof(statistic)}(target, statistic)
end
_base_statistic(s::On) = s.statistic
_bind(target, b::Hard) =
    Hard((_targetid(target), b.id), On(target, b.statistic), b.relation)
_bind(target, b::Approximate) = Approximate(
    (_targetid(target), b.id),
    On(target, b.statistic),
    b.target;
    scale=b.scale,
    importance=b.importance,
)

function _grouped_beliefs(groups)
    derived = filter(p -> p.first isa Derived, groups)
    names = map(p -> p.first.name, derived)
    length(unique(names)) == length(names) ||
        throw(ArgumentError("derived names must be unique"))
    targets = map(p -> _target(p.first), groups)
    length(unique(_targetid.(targets))) == length(targets) ||
        throw(ArgumentError("each target needs one belief collection"))
    Tuple(Iterators.flatten(map(groups, targets) do pair, target
        map(b -> _bind(target, b), Tuple(pair.second))
    end))
end

function _check_derived(beliefs)
    definitions = Dict{Any,Any}()
    for belief in beliefs
        statistic = belief.statistic
        statistic isa On && statistic.target isa Derived || continue
        target = statistic.target
        previous = get!(definitions, target.name, target)
        isequal(previous.references, target.references) && isequal(previous.f, target.f) ||
            throw(ArgumentError("a derived name must keep one definition"))
    end
    nothing
end

struct NoAnalytic <: Exception
    statistic
end
Base.showerror(io::IO, e::NoAnalytic) = print(
    io,
    "No analytic evaluator for ",
    e.statistic,
    "; pass sampling=MonteCarlo(n) to allow sampled evaluation",
)

_law_optic(law, ::VarNames.Iden) = law
function _law_optic(law, optic)
    part = VarNames.ohead(optic)(marginals(law))
    marginal = part isa AbstractArray ? productmeasure(part) : part
    _law_optic(marginal, VarNames.otail(optic))
end
function _law_reference(law, scope, reference)
    scope == reference && return law
    if scope isa Joint
        law isa ProductMeasure &&
            marginals(law) isa NamedTuple &&
            return _law_reference(law, nothing, reference)
        length(scope.references) == 1 &&
            return _law_reference(law, only(scope.references), reference)
        index = findfirst(r -> subsumes(r, reference), scope.references)
        index === nothing && throw(NoAnalytic(reference))
        part = _law_optic(law, VarNames.Index((index,), NamedTuple()))
        return _law_reference(part, scope.references[index], reference)
    end
    if scope === nothing || scope isa Tuple
        return _law_optic(
            getproperty(marginals(law), getsym(reference)),
            getoptic(reference),
        )
    elseif scope isa VarName &&
           getsym(scope) == getsym(reference) &&
           getoptic(scope) isa VarNames.Iden
        return _law_optic(law, getoptic(reference))
    end
    throw(NoAnalytic(reference))
end

function _covers(law, scope, reference::VarName)
    if scope isa VarName
        return subsumes(scope, reference)
    elseif scope isa Joint
        return any(r -> subsumes(r, reference), scope.references)
    elseif scope isa Tuple
        return any(r -> getsym(r) == getsym(reference), scope)
    elseif law isa ProductMeasure && marginals(law) isa NamedTuple
        return haskey(marginals(law), getsym(reference))
    end
    true
end
_applicable(s, law, scope) = true
_applicable(s::On, law, scope) = all(r -> _covers(law, scope, r), _references(s.target))

function _draw_reference(draw, scope, reference)
    if scope === nothing || scope isa Tuple
        return getvalue(draw, reference)
    elseif scope isa VarName
        scope == reference && return draw
        draw isa Number &&
            throw(DimensionMismatch("indexed targets require an array-valued candidate"))
        return getvalue(VarNames.VarNamedTuple((scope=>draw,)), reference)
    else
        draw isa NamedTuple && return getvalue(draw, reference)
        length(scope.references) == 1 &&
            return _draw_reference(draw, only(scope.references), reference)
        index = findfirst(==(reference), scope.references)
        index !== nothing && return draw[index]
        parent = findfirst(r -> subsumes(r, reference), scope.references)
        parent === nothing && throw(KeyError(reference))
        return _draw_reference(draw[parent], scope.references[parent], reference)
    end
end
_draw_target(draw, scope, reference::VarName) = _draw_reference(draw, scope, reference)
_draw_target(draw, scope, target::Joint) =
    map(r -> _draw_reference(draw, scope, r), target.references)
_draw_target(draw, scope, target::Derived) =
    target.f(map(r -> _draw_reference(draw, scope, r), target.references)...)

function _analytic_on(statistic, law, scope, target::VarName)
    evaluate(statistic, _law_reference(law, scope, target))
end
_analytic_on(statistic, law, scope, target::Derived) = throw(NoAnalytic(statistic))
_evaluate(s::On, prior, ::Analytic, rng) =
    _analytic_on(s.statistic, prior, nothing, s.target)
function _analytic_on(statistic, law, scope, target::Joint)
    if scope isa Joint && scope == target
        if statistic isa Mean
            return _estimate(_vector(_draw_target(mean(law), scope, target)))
        elseif statistic isa Std
            return _estimate(sqrt.(diag(_covariance(law))))
        end
        if statistic isa Union{Covariance,Correlation}
            all(x -> x isa Number, mean(law)) ||
                throw(DimensionMismatch("pairwise quantities must be scalar"))
        end
        return evaluate(statistic, law)
    end
    parts = map(r -> _law_reference(law, scope, r), target.references)
    _joint_estimate(statistic, parts, law, scope, target)
end

function _joint_estimate(statistic, parts, law, scope, target)
    if statistic isa Union{Covariance,Correlation}
        all(p -> mean(p) isa Number, parts) ||
            throw(DimensionMismatch("pairwise quantities must be scalar"))
    end
    if _base_statistic(statistic) isa
       Union{Covariance,Correlation,CovarianceMatrix,CorrelationMatrix}
        matrix = _joint_covariance(law, scope, target, parts)
        if statistic isa CovarianceMatrix
            return _estimate(matrix)
        elseif statistic isa CorrelationMatrix
            scales = sqrt.(diag(matrix))
            return _estimate(matrix ./ (scales*scales'))
        end
        size(matrix) == (2, 2) ||
            throw(DimensionMismatch("pairwise statistics require two scalars"))
        value = matrix[1, 2]
        if statistic isa Correlation
            value /= sqrt(matrix[1, 1]*matrix[2, 2])
        end
        return _estimate(value)
    elseif statistic isa Mean
        return _estimate(reduce(vcat, map(p -> _vector(mean(p)), parts)))
    elseif statistic isa Std
        return _estimate(reduce(vcat, map(p -> _vector(std(p)), parts)))
    end
    throw(NoAnalytic(statistic))
end

function _joint_covariance(law, scope, target, parts)
    parents = getsym.(target.references)
    distinct = Tuple(unique(parents))
    if length(distinct) > 1
        independent = scope isa Tuple || (scope === nothing && law isa ProductMeasure)
        independent || throw(NoAnalytic(CovarianceMatrix()))
    end
    blocks = map(distinct) do parent
        parentlaw = _law_reference(law, scope, VarName{parent}())
        center = mean(parentlaw)
        coordinates = center isa Number ? 1 : reshape(1:length(center), size(center))
        indices = reduce(vcat, map(findall(==(parent), parents)) do i
            _vector(getoptic(target.references[i])(coordinates))
        end)
        covariance =
            center isa Number ? reshape([var(parentlaw)], 1, 1) : _covariance(parentlaw)
        covariance[indices, indices]
    end
    lengths = map(p -> length(_vector(mean(p))), parts)
    stops = accumulate(+, lengths)
    positions = map(distinct) do parent
        reduce(vcat, map(findall(==(parent), parents)) do i
            collect((stops[i]-lengths[i]+1):stops[i])
        end)
    end
    T = promote_type(map(eltype, blocks)...)
    matrix = similar(first(blocks), T, last(stops), last(stops))
    fill!(matrix, zero(T))
    for (block, indices) in zip(blocks, positions)
        matrix[indices, indices] = block
    end
    matrix
end
