Base.show(io::IO, mime::MIME"text/plain", report::Union{FitReport,Validation}) =
    _show_report(io, mime, report)
Base.show(io::IO, mime::MIME"text/html", report::Union{FitReport,Validation}) =
    _show_report(io, mime, report)

_display_value(x) = sprint(show, x; context=(:compact=>true, :limit=>true))
_display_value(x::AbstractFloat) = string(round(x; sigdigits=5))
_display_value(::Nothing) = "—"
_display_value(x::Symbol) = string(x)
_display_value(x::Tuple) = "(" * join(map(_display_value, x), ", ") * ")"
function _display_value(x::NamedTuple)
    isempty(x) && return "—"
    join(("$name = $(_display_value(value))" for (name, value) in pairs(x)), ", ")
end
function _display_value(x::AbstractVector)
    length(x) > 8 && return "$(length(x))-element vector"
    "[" * join(map(_display_value, x), ", ") * "]"
end

_display_id(id) = join(id[1:(end-1)], " / ") * " #$(last(id))"
_display_target(target) = string(target)
_display_target(target::Joint) = join(target.references, ", ")
_display_target(target::Union{Derived,Predictive}) = string(target.name)
function _display_belief(belief)
    statistic = belief.statistic
    statistic isa On || return string(belief.id)
    "$(_display_target(statistic.target)) / $(last(belief.id))"
end

_display_judgment(belief::Approximate) = "≈ $(_display_value(belief.target))"
_display_judgment(belief::Hard) = _display_relation(belief.relation)
_display_relation(r::AtLeast) = "≥ $(_display_value(r.value))"
_display_relation(r::AtMost) = "≤ $(_display_value(r.value))"
_display_relation(r::EqualTo) = "= $(_display_value(r.value))"
_display_relation(r::Between) = "[$(_display_value(r.lower)), $(_display_value(r.upper))]"

function _display_estimate(estimate)
    estimate === nothing && return "—"
    value = _display_value(estimate.value)
    isequal(estimate.lower, estimate.upper) && return value
    "$value [$(_display_value(estimate.lower)), $(_display_value(estimate.upper))]"
end

function _display_evidence(estimate)
    estimate === nothing && return "—"
    estimate.n == 0 && return string(estimate.evidence)
    "$(estimate.evidence) (n=$(estimate.n))"
end

function _show_report(io, mime, report::FitReport)
    rows = [
        (
            _display_id(result.id),
            string(_fit_status(result)),
            _display_value(result.parameters),
        ) for result in report.results
    ]
    _show_report_table(
        io,
        mime,
        "Candidate fits",
        ("Candidate", "Status", "Parameters"),
        rows,
    )
end

function _show_report(io, mime, validation::Validation)
    report = _fit_report(validation.report)
    rows = [
        (
            _display_id(result.id),
            _display_belief(belief),
            _display_judgment(belief),
            _display_estimate(check.estimate),
            _display_evidence(check.estimate),
            string(check.status),
        ) for (result, checks) in zip(report.results, validation.checks) for
        (belief, check) in zip(report.problem.beliefs, checks)
    ]
    headers = ("Candidate", "Belief", "Judgment", "Estimate", "Evidence", "Status")
    _show_report_table(io, mime, "Belief checks", headers, rows)
end

function _show_report_table(io, ::MIME"text/plain", title, headers, rows)
    n = length(rows)
    if get(io, :limit, false)
        rows = first(rows, min(n, max(1, first(displaysize(io)) - 4)))
    end
    widths = map(eachindex(headers)) do i
        max(textwidth(headers[i]), maximum(row -> textwidth(row[i]), rows; init=0))
    end
    print(io, title)
    for row in Iterators.flatten(((headers,), rows))
        line = join((rpad(row[i], widths[i]) for i in eachindex(widths)), "  ")
        print(io, '\n', rstrip(line))
    end
    length(rows) < n && print(io, "\n… $(n-length(rows)) more rows")
end

_escape_report_html(s) = replace(s, '&'=>"&amp;", '<'=>"&lt;", '>'=>"&gt;")
function _show_report_table(io, ::MIME"text/html", title, headers, rows)
    print(io, "<table><caption>", _escape_report_html(title), "</caption><thead><tr>")
    for name in headers
        print(io, "<th scope=\"col\">", _escape_report_html(name), "</th>")
    end
    print(io, "</tr></thead><tbody>")
    for row in rows
        print(io, "<tr>")
        for value in row
            print(io, "<td>", _escape_report_html(value), "</td>")
        end
        print(io, "</tr>")
    end
    print(io, "</tbody></table>")
end
