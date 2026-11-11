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
struct Validation{R,C,P,S}
    report::R
    checks::C
    plan::P
    seed::S
end
Validation(report, checks, plan) = Validation(report, checks, plan, nothing)

function _check_estimate(belief, estimate)
    estimate === nothing && return Check(belief.id, nothing, :pending, nothing)
    estimate isa EvaluationFailure &&
        return Check(belief.id, nothing, :undefined, estimate.error)
    try
        status = if belief isa Approximate
            _discrepancy(belief, estimate.value)
            :target
        else
            entries = _hard_entries(belief, estimate.value)
            isempty(entries) &&
                throw(ArgumentError("hard belief has no supplied free constraints"))
            satisfied = all(entries) do index
                lower, upper = _entry_bounds(belief, index)
                lower <= _at(estimate.lower, index) &&
                    _at(estimate.upper, index) <= upper
            end
            violated = any(entries) do index
                lower, upper = _entry_bounds(belief, index)
                _at(estimate.upper, index) < lower || upper < _at(estimate.lower, index)
            end
            if satisfied
                :satisfied
            elseif violated
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
    validate(rng, report; plan=Analytic(), sampling=report.sampling)

Reevaluate fixed results with fresh draws. The default inherits the fitting sampling plan.
Pass `sampling=nothing` for analytic checks. Monte Carlo error is shared across all checks.
"""
function validate(
    rng::Union{AbstractRNG,GenContext},
    report::FitReport;
    plan=Analytic(),
    sampling=report.sampling,
)
    _validation(rng, report, plan; sampling)
end
function _validation(rng, report, plan; sampling=nothing)
    force_sampling = plan isa MonteCarlo
    budget = force_sampling ? plan : sampling
    # Each array evaluator divides its check budget across its coordinates.
    nchecks = length(report.problem.beliefs) * length(report.results)
    per_check = if budget isa MonteCarlo && nchecks > 0
        MonteCarlo(budget.n; alpha=budget.alpha/nchecks)
    else
        budget
    end
    seed = budget === nothing ? nothing : _sampling_seed(rng)
    stream = seed === nothing ? rng : _sampling_stream(rng, seed)
    checks = map(report.results) do result
        if result.error === nothing
            estimates = _estimates(
                report.problem.beliefs,
                result.measure,
                result.candidate.scope;
                sampling=per_check,
                rng=stream,
                force_sampling,
            )
            map(_check_estimate, report.problem.beliefs, estimates)
        else
            map(b -> Check(b.id, nothing, :undefined, result.error), report.problem.beliefs)
        end
    end
    Validation(report, checks, budget === nothing ? plan : budget, seed)
end

struct Selection{R,V,I,N}
    report::R
    validation::V
    index::I
    role::Symbol
    rationale::N
end
_factor_scope(selection::Selection) =
    selection.report.results[selection.index].candidate.scope
_factor_length(::Selection) = 0
_factor_law(selection::Selection, parameters) = measure(selection)
function _requirements(selection::Selection)
    result = selection.report.results[selection.index]
    filter(
        b -> b isa Hard && _applicable(b.statistic, result.measure, result.candidate.scope),
        selection.report.problem.beliefs,
    )
end

"""`unlock(selection)` returns its fitted family start while retaining the selected law's hard beliefs."""
function unlock(selection::Selection)
    result = selection.report.results[selection.index]
    builder = result.candidate.build
    original = builder isa NamedBuilder ? builder.f : builder
    Candidate(
        result.candidate.id,
        original,
        (result.parameters,);
        lower=result.candidate.lower,
        upper=result.candidate.upper,
        scope=result.candidate.scope,
        requirements=_requirements(selection),
    )
end

function _acceptable(candidate, checks)
    local_scope = candidate.scope isa Union{VarName,Joint}
    all(
        c -> c.status in (:satisfied, :target) || (local_scope && c.status == :pending),
        checks,
    )
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
        validation isa Validation || throw(ArgumentError("acceptance requires validation"))
        validation.report === report ||
            throw(ArgumentError("validation belongs to another report"))
        _acceptable(result.candidate, validation.checks[index]) || throw(
            ArgumentError("hard requirements remain violated, undefined, or inconclusive"),
        )
    end
    Selection(report, validation, index, role, rationale)
end

function select(report::FitReport, target, result_id; kwargs...)
    reference = _targetid(_target(target))
    select(report, (reference, result_id...); kwargs...)
end

"""
    measure(selection)

Return the selected native MeasureBase measure, separate from its fitting report.
"""
measure(selection::Selection) = selection.report.results[selection.index].measure

"""
    compare(report; validation=nothing)
    compare(rng, report; plan=Analytic(), sampling=report.sampling)

Compare feasible results using simultaneous discrepancy bounds. Sampled reports require
supplied validation or an RNG for fresh validation. Estimated-only evidence stays unresolved.
"""
function compare(report::FitReport; validation=nothing)
    if validation === nothing
        report.sampling === nothing || throw(
            ArgumentError(
                "sampled comparison requires compare(rng, report) or supplied validation",
            ),
        )
        validation = _validation(nothing, report, Analytic())
    end
    validation.report === report ||
        throw(ArgumentError("comparison requires validation of this report"))
    _compare(report.problem.beliefs, collect(report.results), collect(validation.checks))
end

function compare(rng::Union{AbstractRNG,GenContext}, report::FitReport; kwargs...)
    compare(report; validation=validate(rng, report; kwargs...))
end

function _discrepancy_bounds(belief, estimate)
    if estimate.lower == estimate.upper
        value = sqrt(_discrepancy(belief, estimate.value))
        return (value, value)
    end
    bounds = map(belief.entries) do index
        target = _at(belief.statistic, belief.target, index)
        scale = _at(belief.statistic, belief.scale, index)
        lower = (_at(belief.statistic, estimate.lower, index) - target) / scale
        upper = (_at(belief.statistic, estimate.upper, index) - target) / scale
        nearest = min(abs(lower), abs(upper))
        if lower <= 0 <= upper
            nearest = zero(nearest)
        end
        (nearest, max(abs(lower), abs(upper)))
    end
    (
        sqrt(sum(x -> abs2(first(x)), bounds) / length(bounds)),
        sqrt(sum(x -> abs2(last(x)), bounds) / length(bounds)),
    )
end

function _compare(beliefs, results, checks)
    feasible = findall(eachindex(results)) do i
        results[i].error === nothing && _acceptable(results[i].candidate, checks[i])
    end
    ids = [results[i].id for i in feasible]
    isempty(ids) &&
        return (; feasible=ids, dominates=[], ties=[], incomparable=[], unresolved=[])
    soft = findall(b -> b isa Approximate, beliefs)
    ranges = map(feasible) do i
        map(soft) do j
            check = checks[i][j]
            if check.status == :pending
                return nothing
            end
            _discrepancy_bounds(beliefs[j], check.estimate)
        end
    end
    estimated = map(feasible) do i
        any(soft) do j
            estimate = checks[i][j].estimate
            estimate !== nothing &&
                !(estimate.evidence in (:analytic, :certified, :statistical))
        end
    end
    dominates, ties, incomparable, unresolved =
        ntuple(_ -> Tuple{eltype(ids),eltype(ids)}[], 4)
    for i in eachindex(feasible), j in (i+1):length(feasible)
        pair = (ids[i], ids[j])
        results[feasible[i]].candidate.scope == results[feasible[j]].candidate.scope ||
            continue
        map(isnothing, ranges[i]) == map(isnothing, ranges[j]) || continue
        left = filter(!isnothing, ranges[i])
        right = filter(!isnothing, ranges[j])
        if estimated[i] || estimated[j]
            push!(unresolved, pair)
        elseif all(x -> first(x) == last(x), left) && left == right
            push!(ties, pair)
        elseif _dominates_bounds(left, right)
            push!(dominates, pair)
        elseif _dominates_bounds(right, left)
            push!(dominates, reverse(pair))
        elseif any(last(a) < first(b) for (a, b) in zip(left, right)) &&
               any(last(b) < first(a) for (a, b) in zip(left, right))
            push!(incomparable, pair)
        else
            push!(unresolved, pair)
        end
    end
    (; feasible=ids, dominates, ties, incomparable, unresolved)
end

_dominates_bounds(a, b) =
    all(last(x) <= first(y) for (x, y) in zip(a, b)) &&
    any(last(x) < first(y) for (x, y) in zip(a, b))
