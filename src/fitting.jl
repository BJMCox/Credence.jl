"""
    Candidate(id, build, starts; lower=nothing, upper=nothing)

A MeasureBase family builder with explicit starts and optional parameter bounds.
"""
struct Candidate{I,F,S,L,U,T,B}
    id::I
    build::F
    starts::S
    lower::L
    upper::U
    scope::T
    requirements::B
end

function Candidate(
    id,
    build,
    starts;
    lower=nothing,
    upper=nothing,
    scope=nothing,
    requirements=(),
)
    points = if starts isa NamedTuple
        (starts,)
    elseif starts isa AbstractVector{<:Number}
        (Tuple(starts),)
    else
        Tuple(p isa NamedTuple ? p : Tuple(p) for p in starts)
    end
    isempty(points) && throw(ArgumentError("at least one parameter start is required"))
    n = length(_flatten(first(points)))
    all(p -> _layout(p) == _layout(first(points)), points) ||
        throw(ArgumentError("parameter starts must have equal lengths"))
    lo = lower === nothing ? nothing : Tuple(_flatten_bound(lower, first(points)))
    hi = upper === nothing ? nothing : Tuple(_flatten_bound(upper, first(points)))
    for bound in (lo, hi)
        bound === nothing ||
            length(bound) == n ||
            throw(ArgumentError("parameter bounds have the wrong length"))
    end
    if lo !== nothing && hi !== nothing
        all(lo .<= hi) || throw(ArgumentError("lower parameter bounds exceed upper bounds"))
    end
    builder = first(points) isa NamedTuple ? NamedBuilder(build, first(points)) : build
    Candidate(id, builder, points, lo, hi, scope, requirements)
end

_flatten(x::Number) = [x]
_flatten(x::AbstractArray) = vec(x)
_flatten(x::Union{Tuple,NamedTuple}) =
    isempty(x) ? Float64[] : reduce(vcat, map(_flatten, values(x)))
_flatten_bound(bound, template) = _flatten(bound)
function _flatten_bound(bound::AbstractArray, template::AbstractArray)
    axes(bound) == axes(template) ||
        throw(DimensionMismatch("parameter bound axes must match the starts"))
    vec(bound)
end
_flatten_bound(bound::Number, template::AbstractArray) = fill(bound, length(template))
function _flatten_bound(bound::NamedTuple, template::NamedTuple)
    Set(keys(bound)) == Set(keys(template)) ||
        throw(ArgumentError("parameter bound names must match the starts"))
    reduce(
        vcat,
        map(
            k -> _flatten_bound(getproperty(bound, k), getproperty(template, k)),
            keys(template),
        ),
    )
end
_layout(x::Number) = nothing
_layout(x::AbstractArray) = axes(x)
_layout(x::Tuple) = map(_layout, x)
_layout(x::NamedTuple) = (keys(x), map(_layout, values(x)))
function _restore(template::Number, values, offset)
    offset[] += 1
    values[offset[]]
end
function _restore(template::AbstractArray, values, offset)
    indices = (offset[]+1):(offset[]+length(template))
    offset[] += length(template)
    reshape(view(values, indices), size(template))
end
_restore(template::NamedTuple, values, offset) =
    NamedTuple{keys(template)}(map(x -> _restore(x, values, offset), Base.values(template)))
_restore(template::Tuple, values, offset) = map(x -> _restore(x, values, offset), template)
struct NamedBuilder{F,T}
    f::F
    template::T
end
(build::NamedBuilder)(values::AbstractVector) =
    build.f(_restore(build.template, values, Ref(0)))
(build::NamedBuilder)(values::NamedTuple) = build.f(values)
_parameters(candidate, values) = Tuple(values)
_parameters(candidate::Candidate{I,<:NamedBuilder}, values) where {I} =
    _restore(candidate.build.template, values, Ref(0))
_scoped(candidate, target) = Candidate(
    candidate.id,
    candidate.build,
    candidate.starts,
    candidate.lower,
    candidate.upper,
    target,
    candidate.requirements,
)
_resultid(candidate, index) =
    candidate.scope === nothing || candidate.scope isa Tuple ? (candidate.id, index) :
    (_targetid(candidate.scope), candidate.id, index)

"""
    Elicitation(beliefs, candidates)

An immutable collection of beliefs and candidate definitions.
"""
struct Elicitation{B,C}
    beliefs::B
    candidates::C
    function Elicitation(beliefs, candidates; combinations=nothing)
        rawbeliefs, rawcandidates = Tuple(beliefs), Tuple(candidates)
        grouped = !isempty(rawbeliefs) && first(rawbeliefs) isa Pair
        bs = grouped ? _grouped_beliefs(rawbeliefs) : rawbeliefs
        catalog = !isempty(rawcandidates) && first(rawcandidates) isa Pair
        cs = catalog ? _grouped_candidates(rawcandidates) : rawcandidates
        if combinations !== nothing
            cs = _combinations(cs, combinations)
        end
        bs = _inherit_requirements(bs, cs)
        _check_derived(bs)
        length(unique(b.id for b in bs)) == length(bs) ||
            throw(ArgumentError("belief IDs must be unique"))
        length(unique(_resultid(c, 1) for c in cs)) == length(cs) ||
            throw(ArgumentError("candidate IDs must be unique"))
        new{typeof(bs),typeof(cs)}(bs, cs)
    end
end

"""`replace(problem, target => belief)` replaces one locally named belief, preserving the original problem."""
function Base.replace(problem::Elicitation, change::Pair)
    target = _target(change.first)
    belief = _bind(target, change.second)
    found = any(b -> b.id == belief.id, problem.beliefs)
    found || throw(ArgumentError("no belief with this target and ID exists"))
    beliefs = map(problem.beliefs) do old
        if old.id == belief.id
            return belief
        end
        old
    end
    Elicitation(beliefs, problem.candidates)
end

function Base.getindex(problem::Elicitation, target, id)
    identity = (_targetid(_target(target)), id)
    only(filter(b -> b.id == identity, problem.beliefs))
end

"""
    Evaluate()

Evaluate each declared parameter point as a separate candidate result.
"""
struct Evaluate end

"""
    FitResult

One retained parameter point and its native measure, solver result, or failure.
"""
struct FitResult{I,C,P,M,S,E}
    id::I
    candidate::C
    parameters::P
    measure::M
    solver::S
    error::E
end

"""
    FitReport

All retained candidate results for one elicitation problem.
"""
mutable struct FitReport{P,R,S,N}
    const problem::P
    const results::R
    const sampling::S
    const seed::N
end
FitReport(problem, results) = FitReport(problem, results, nothing, nothing)

"""
    diagnostics(problem_or_report)

Report matrix inconsistencies and conflicts between ordered quantiles or nested event
probabilities. Each row contains `id` and `code`. A cross-belief conflict identifies
the pair of belief IDs. The submitted judgments remain unchanged.
"""
function diagnostics(problem::Elicitation)
    local_checks = Tuple(
        (id=b.id, code=code) for b in problem.beliefs if b isa Approximate for
        code in b.diagnostics
    )
    conflicts = []
    beliefs = problem.beliefs
    for i in eachindex(beliefs), j in (i+1):length(beliefs)
        a, b = beliefs[i], beliefs[j]
        _judgment_scope(a.statistic) == _judgment_scope(b.statistic) || continue
        sa, sb = _base_statistic(a.statistic), _base_statistic(b.statistic)
        code = if sa isa Quantile && sb isa Quantile
            :inconsistent_quantiles
        elseif sa isa Probability{<:IntervalEvent} && sb isa Probability{<:IntervalEvent}
            :inconsistent_probabilities
        else
            continue
        end
        if (_statistic_order(sa, sb) && _judgment_conflict(a, b)) ||
           (_statistic_order(sb, sa) && _judgment_conflict(b, a))
            push!(conflicts, (id=(a.id, b.id), code=code))
        end
    end
    (local_checks..., conflicts...)
end
diagnostics(report::FitReport) = diagnostics(report.problem)

_judgment_scope(s) = nothing
_judgment_scope(s::On) = _targetid(s.target)
_statistic_order(a::Quantile, b::Quantile) = a.p <= b.p
_statistic_order(a::Probability, b::Probability) =
    b.event.lower <= a.event.lower && a.event.upper <= b.event.upper
_judgment_bounds(b::Approximate) = (b.target, b.target)
_judgment_bounds(b::Hard) = _bounds(b.relation)
function _judgment_conflict(a, b)
    lower, _ = _judgment_bounds(a)
    _, upper = _judgment_bounds(b)
    if lower isa AbstractArray && upper isa AbstractArray && axes(lower) != axes(upper)
        return false
    end
    template = lower
    if upper isa AbstractArray
        template = upper
    end
    any(_indices(template)) do index
        lo, hi = _at(lower, index), _at(upper, index)
        !_absent(lo) && !_absent(hi) && lo > hi
    end
end

function _result(candidate, index, parameters, prior, solver)
    if candidate.lower !== nothing
        all(parameters .>= candidate.lower) ||
            throw(ArgumentError("parameters lie below the candidate domain"))
    end
    if candidate.upper !== nothing
        all(parameters .<= candidate.upper) ||
            throw(ArgumentError("parameters lie above the candidate domain"))
    end
    prior isa MeasureBase.AbstractMeasure ||
        throw(ArgumentError("candidate builder must return a MeasureBase measure"))
    isone(MeasureBase.massof(prior)) ||
        throw(ArgumentError("candidate measure must be normalized"))
    FitResult(
        _resultid(candidate, index),
        candidate,
        _parameters(candidate, parameters),
        prior,
        solver,
        nothing,
    )
end
_failure(candidate, index, error) = FitResult(
    _resultid(candidate, index),
    candidate,
    candidate.starts[index],
    nothing,
    nothing,
    error,
)

"""
    fit(rng, problem; plan=Evaluate())

Evaluate declared parameter points, or fit through an optional optimizer extension.
"""
function fit(
    rng::Union{AbstractRNG,GenContext},
    problem::Elicitation;
    plan=Evaluate(),
    sampling=nothing,
)
    _fit(rng, problem, plan; sampling)
end
function _fit(rng, problem, ::Evaluate; sampling=nothing)
    results = map(problem.candidates) do candidate
        map(Tuple(enumerate(candidate.starts))) do (index, point)
            try
                values = copy(_flatten(point))
                _result(candidate, index, values, candidate.build(values), nothing)
            catch error
                error isa InterruptException && rethrow()
                _failure(candidate, index, error)
            end
        end
    end
    FitReport(problem, Tuple(Iterators.flatten(results)), sampling, nothing)
end

function _loss(problem, prior; scope=nothing, sampling=nothing, rng=nothing)
    soft = filter(b -> b isa Approximate, problem.beliefs)
    isempty(soft) && return 0.0
    weights = map(soft) do belief
        weight = something(belief.importance, 1)
        _applicable(belief.statistic, prior, scope) ? weight : zero(weight)
    end
    maxweight = maximum(weights)
    iszero(maxweight) && return 0.0
    estimates = if sampling === nothing
        map(soft, weights) do belief, weight
            iszero(weight) ? nothing : _analytic(belief.statistic, prior, scope)
        end
    else
        _estimates(soft, prior, scope; sampling, rng)
    end
    return sum(map(soft, weights, estimates) do belief, weight, estimate
        iszero(weight) && return 0.0
        estimate isa EvaluationFailure && throw(estimate.error)
        discrepancy = _discrepancy(belief, estimate.value)
        oftype(discrepancy, weight / maxweight) * discrepancy
    end)
end
