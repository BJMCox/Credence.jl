"A MeasureBase family builder with explicit starts and optional parameter bounds."
struct Candidate{I,F,S,L,U}
    id::I
    build::F
    starts::S
    lower::L
    upper::U
end
function Candidate(id, build, starts; lower=nothing, upper=nothing)
    points = if starts isa AbstractVector{<:Number}
        (Tuple(starts),)
    else
        Tuple(Tuple(p) for p in starts)
    end
    isempty(points) && throw(ArgumentError("at least one parameter start is required"))
    n = length(first(points))
    all(p -> length(p) == n, points) ||
        throw(ArgumentError("parameter starts must have equal lengths"))
    lo = lower === nothing ? nothing : Tuple(lower)
    hi = upper === nothing ? nothing : Tuple(upper)
    for bound in (lo, hi)
        bound === nothing ||
            length(bound) == n ||
            throw(ArgumentError("parameter bounds have the wrong length"))
    end
    if lo !== nothing && hi !== nothing
        all(lo .<= hi) || throw(ArgumentError("lower parameter bounds exceed upper bounds"))
    end
    Candidate(id, build, points, lo, hi)
end

"An immutable collection of beliefs and candidate definitions."
struct Elicitation{B,C}
    beliefs::B
    candidates::C
    function Elicitation(beliefs, candidates)
        bs, cs = Tuple(beliefs), Tuple(candidates)
        length(unique(b.id for b in bs)) == length(bs) ||
            throw(ArgumentError("belief IDs must be unique"))
        length(unique(c.id for c in cs)) == length(cs) ||
            throw(ArgumentError("candidate IDs must be unique"))
        new{typeof(bs),typeof(cs)}(bs, cs)
    end
end

"Evaluate the declared parameter points without optimizing or assigning probability weights."
struct Evaluate end

"One retained parameter point and its native measure, solver result, or failure."
struct FitResult{I,C,P,M,S,E}
    id::I
    candidate::C
    parameters::P
    measure::M
    solver::S
    error::E
end

"All retained candidate results for one elicitation problem."
struct FitReport{P,R}
    problem::P
    results::R
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
    FitResult((candidate.id, index), candidate, Tuple(parameters), prior, solver, nothing)
end
_failure(candidate, index, error) = FitResult(
    (candidate.id, index),
    candidate,
    candidate.starts[index],
    nothing,
    nothing,
    error,
)

"Evaluate declared parameter points, or fit through an optional optimizer extension."
function fit(rng::Union{AbstractRNG,GenContext}, problem::Elicitation; plan=Evaluate())
    _fit(rng, problem, plan)
end
function _fit(rng, problem, ::Evaluate)
    results = map(problem.candidates) do candidate
        map(Tuple(enumerate(candidate.starts))) do (index, point)
            try
                _result(candidate, index, point, candidate.build(collect(point)), nothing)
            catch error
                error isa InterruptException && rethrow()
                _failure(candidate, index, error)
            end
        end
    end
    FitReport(problem, Tuple(Iterators.flatten(results)))
end

function _loss(problem, prior)
    soft = filter(b -> b isa Approximate, problem.beliefs)
    isempty(soft) && return 0.0
    if length(soft) > 1 && any(b -> b.importance === nothing, soft)
        throw(
            ArgumentError(
                "multiple soft targets require explicit importance weights for scalar optimization; compare evaluated points for Pareto ranking",
            ),
        )
    end
    weights = map(b -> something(b.importance, 1), soft)
    maxweight = maximum(weights)
    iszero(maxweight) && return 0.0
    return sum(
        map(soft, weights) do belief, weight
            iszero(weight) && return 0.0
            residual =
                (evaluate(belief.statistic, prior).value-belief.target) / belief.scale
            (weight / maxweight) * residual^2
        end;
        init=0.0,
    )
end
