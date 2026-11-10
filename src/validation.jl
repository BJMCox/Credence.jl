"""
    Check

A belief's evaluated value, uncertainty, status, and any evaluation error.
"""
struct Check{I,E,X}
    id::I
    estimate::E
    status::Symbol
    error::X
end

"""
    Validation

Fresh belief checks tied to one fit report.
"""
struct Validation{R,C,P}
    report::R
    checks::C
    plan::P
end

function _check(belief, prior, plan, rng)
    try
        estimate = evaluate(belief.statistic, prior; plan, rng)
        status = if belief isa Approximate
            :target
        else
            lower, upper = _bounds(belief.relation)
            if lower <= estimate.lower && estimate.upper <= upper
                :satisfied
            elseif estimate.upper < lower || upper < estimate.lower
                :violated
            else
                :inconclusive
            end
        end
        if status in (:satisfied, :violated) &&
           !(estimate.evidence in (:analytic, :certified, :statistical))
            status = :inconclusive
        end
        Check(belief.id, estimate, status, nothing)
    catch error
        error isa InterruptException && rethrow()
        Check(belief.id, nothing, :undefined, error)
    end
end

"""
    validate(rng, report; plan=Analytic())

Reevaluate fixed results. `MonteCarlo` allocates alpha across all hard checks in this report.
"""
function validate(rng::Union{AbstractRNG,GenContext}, report::FitReport; plan=Analytic())
    _validation(rng, report, plan)
end
function _validation(rng, report, plan)
    nchecks = count(b -> b isa Hard, report.problem.beliefs) * length(report.results)
    per_check = if plan isa MonteCarlo && nchecks > 0
        MonteCarlo(plan.n; alpha=plan.alpha/nchecks)
    else
        plan
    end
    checks = map(report.results) do result
        if result.error === nothing
            map(b -> _check(b, result.measure, per_check, rng), report.problem.beliefs)
        else
            map(b -> Check(b.id, nothing, :undefined, result.error), report.problem.beliefs)
        end
    end
    Validation(report, checks, plan)
end

struct Selection{R,V,I,N}
    report::R
    validation::V
    index::I
    role::Symbol
    rationale::N
end

"""
    select(report, result_id; role=:accepted, validation=nothing, rationale)

Select one concrete result as a reference or accepted prior, preserving its validation.
"""
function select(report::FitReport, result_id; role=:accepted, validation=nothing, rationale)
    role in (:reference, :accepted) ||
        throw(ArgumentError("selection role must be :reference or :accepted"))
    index = findfirst(r -> r.id == result_id, report.results)
    index === nothing && throw(ArgumentError("unknown result ID"))
    result = report.results[index]
    result.error === nothing || throw(ArgumentError("cannot select a failed candidate"))
    if role == :accepted
        validation isa Validation && validation.report === report ||
            throw(ArgumentError("acceptance requires validation of this report"))
        all(c -> c.status in (:satisfied, :target), validation.checks[index]) || throw(
            ArgumentError("hard requirements remain violated, undefined, or inconclusive"),
        )
    end
    Selection(report, validation, index, role, rationale)
end

"""
    measure(selection)

Return the selected native MeasureBase measure, separate from its fitting report.
"""
measure(selection::Selection) = selection.report.results[selection.index].measure

"""
    compare(report)

Compare feasible results by per-belief discrepancies, retaining ties and incomparability.
"""
function compare(report::FitReport)
    checks = collect(_validation(nothing, report, Analytic()).checks)
    return _compare(report.problem.beliefs, collect(report.results), checks)
end

function _compare(beliefs, results, checks)
    feasible = findall(eachindex(results)) do i
        results[i].error === nothing &&
            all(c -> c.status in (:satisfied, :target), checks[i])
    end
    ids = [results[i].id for i in feasible]
    isempty(ids) &&
        return (; feasible=ids, dominates=[], ties=[], incomparable=[], unresolved=[])
    soft = findall(b -> b isa Approximate, beliefs)
    scores = map(feasible) do i
        [abs(checks[i][j].estimate.value-beliefs[j].target)/beliefs[j].scale for j in soft]
    end
    uncertain = map(feasible) do i
        any(soft) do j
            estimate = checks[i][j].estimate
            estimate.lower != estimate.upper ||
                !(estimate.evidence in (:analytic, :certified, :statistical))
        end
    end
    dominates, ties, incomparable, unresolved =
        ntuple(_ -> Tuple{eltype(ids),eltype(ids)}[], 4)
    for i in eachindex(feasible), j in (i+1):length(feasible)
        pair = (ids[i], ids[j])
        if uncertain[i] || uncertain[j]
            push!(unresolved, pair)
        elseif scores[i] == scores[j]
            push!(ties, pair)
        elseif _dominates(scores[i], scores[j])
            push!(dominates, pair)
        elseif _dominates(scores[j], scores[i])
            push!(dominates, reverse(pair))
        else
            push!(incomparable, pair)
        end
    end
    (; feasible=ids, dominates, ties, incomparable, unresolved)
end

_dominates(a, b) = all(x <= y for (x, y) in zip(a, b)) && any(x < y for (x, y) in zip(a, b))
