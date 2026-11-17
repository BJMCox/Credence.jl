"""
    save_session(path, report::TiltReport; validation=nothing, selections=(), kwargs...)
    save_session(path, selection; kwargs...)

Save a fitted projection or a selected prior with its report and validation.
Fitted measures and projection evidence are restored without optimization.
"""
save_session(path, report::TiltReport; kwargs...) =
    save_session(path, report.fit.problem; report, kwargs...)
save_session(path, selection::Selection; kwargs...) = save_session(
    path,
    _fit_report(selection.report).problem;
    report=selection.report,
    validation=selection.validation,
    selections=(selection,),
    kwargs...,
)

_session_pack(x::Returns, callbacks) =
    _session_tag("returns"; value=_session_pack(x.value, callbacks))
_session_pack(::typeof(identity), callbacks) = _session_tag("identity_function")
_session_pack(x::MeasureBase.AsMeasure, callbacks) =
    _session_tag("asmeasure"; value=_session_pack(x.obj, callbacks))
_session_pack(x::TiltKernel, callbacks) = _session_tag(
    "tilt_kernel";
    score=_session_pack(_tilt_score(x), callbacks),
    base=_session_pack(x.base, callbacks),
)

const _session_measures = (
    MeasureBase.StdNormal,
    MeasureBase.StdUniform,
    MeasureBase.StdExponential,
    MeasureBase.Dirac,
    MeasureBase.WeightedMeasure,
    MeasureBase.NormalizedMeasure,
    MeasureBase.SuperpositionMeasure,
    MeasureBase.TruncatedMeasure,
    MeasureBase.ProductMeasure,
    MeasureBase.UnknownFiniteMass,
    MeasureBase.UnknownMass,
)
_session_pack(x::Union{_session_measures...}, callbacks) = _session_tag(
    "measure";
    name=string(nameof(typeof(x))),
    fields=[_session_pack(getfield(x, i), callbacks) for i in 1:fieldcount(typeof(x))],
)
function _session_unpack_measure(data, callbacks)
    name=Symbol(data["name"])
    index=findfirst(T -> nameof(T)==name, _session_measures)
    index===nothing && throw(ArgumentError("unsupported saved measure $name"))
    _session_measures[index](map(v -> _session_unpack(v, callbacks), data["fields"])...)
end

function _session_pack_report(report::TiltReport, callbacks)
    _session_tag(
        "tilt_report";
        reference=_session_pack(report.reference, callbacks),
        method=_session_pack(report.method, callbacks),
        fit=_session_pack_report(report.fit, callbacks),
        normalization=_session_pack(report.normalization, callbacks),
        divergence=_session_pack(report.divergence, callbacks),
        ratio_bounds=_session_pack(report.ratio_bounds, callbacks),
        feasibility=string(report.feasibility),
        optimality=string(report.optimality),
    )
end
function _session_unpack_tilt_report(data, problem, callbacks)
    TiltReport(
        _session_unpack(data["reference"], callbacks),
        _session_unpack(data["method"], callbacks),
        _session_unpack_report(data["fit"], problem, callbacks),
        _session_unpack(data["normalization"], callbacks),
        _session_unpack(data["divergence"], callbacks),
        _session_unpack(data["ratio_bounds"], callbacks),
        Symbol(data["feasibility"]),
        Symbol(data["optimality"]),
    )
end
