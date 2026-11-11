"""
    fit_summary(report)

Return one named row per fit result, containing its ID, fitted parameters, solver
status, and error. Results evaluated without optimization have status `:evaluated`.
"""
fit_summary(report::FitReport) = map(report.results) do result
    (;
        id=result.id,
        parameters=result.parameters,
        status=_fit_status(result),
        error=result.error,
    )
end

function _fit_status(result)
    result.error === nothing || return :failed
    result.solver === nothing && return :evaluated
    result.solver.retcode
end

"""
    belief_summary(validation; result_id=nothing)

Return one named row per retained belief check. Rows contain result and belief IDs,
targets or bounds, achieved values and enclosures, evidence, status, RMS scaled
approximate-target discrepancy, and errors. Set `result_id` to restrict the rows.
"""
function belief_summary(validation::Validation; result_id=nothing)
    report = validation.report
    [
        _belief_row(result.id, belief, check) for
        (result, checks) in zip(report.results, validation.checks) if
        result_id === nothing || result.id == result_id for
        (belief, check) in zip(report.problem.beliefs, checks)
    ]
end
_summary_target(belief::Approximate) = belief.target
_summary_target(belief::Hard) = _bounds(belief.relation)
function _belief_row(result_id, belief, check)
    estimate = check.estimate
    value, lower, upper, evidence = nothing, nothing, nothing, nothing
    discrepancy = nothing
    if estimate !== nothing
        value, lower, upper, evidence =
            estimate.value, estimate.lower, estimate.upper, estimate.evidence
        if belief isa Approximate
            discrepancy = sqrt(_discrepancy(belief, value))
        end
    end
    (;
        result_id,
        belief_id=belief.id,
        target=_summary_target(belief),
        value,
        lower,
        upper,
        evidence,
        status=check.status,
        discrepancy,
        error=check.error,
    )
end

function Base.show(io::IO, problem::Elicitation)
    print(
        io,
        "Elicitation(",
        length(problem.beliefs),
        " beliefs, ",
        length(problem.candidates),
        " candidates)",
    )
end
function Base.show(io::IO, report::FitReport)
    failures = count(result -> result.error !== nothing, report.results)
    print(io, "FitReport(", length(report.results), " results, ", failures, " failed)")
end
function Base.show(io::IO, result::FitResult)
    print(io, "FitResult(")
    show(io, result.id)
    print(io, ", ", _fit_status(result), ")")
end
function Base.show(io::IO, validation::Validation)
    checks = Iterators.flatten(validation.checks)
    states = (:satisfied, :target, :violated, :inconclusive, :undefined, :pending)
    counts = [
        string(count(check -> check.status == state, checks), " ", state) for
        state in states if any(check -> check.status == state, checks)
    ]
    print(io, "Validation(", length(validation.checks), " results")
    isempty(counts) || print(io, "; ", join(counts, ", "))
    print(io, ")")
end
function Base.show(io::IO, selection::Selection)
    print(io, "Selection(")
    show(io, selection.report.results[selection.index].id)
    print(io, ", ", selection.role, ")")
end
Base.show(
    io::IO,
    ::MIME"text/plain",
    value::Union{Elicitation,FitReport,FitResult,Validation,Selection},
) = show(io, value)
