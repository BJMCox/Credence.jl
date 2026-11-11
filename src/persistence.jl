"""
    save_session(path, problem; report=nothing, validation=nothing,
                 selections=(), history=(), metadata=(;), callbacks=(;))

Save judgments, family definitions, fitted parameters, validation, and selection
rationales as versioned TOML. History can contain earlier problems and plain records.
Register custom builders and statistical, derived, or predictive callbacks under stable
names. Entries can be functions or `(callback=f, version="1")` records. Source-backed
callbacks record source and capture fingerprints. REPL callbacks and opaque captures
require an explicit version. Loading verifies fingerprints before restoring validation.
Direct global data is fingerprinted too. Versions are the caller's reconstruction contract
for transitive helpers and opaque state, which cannot be checked automatically.
"""
function save_session(
    path,
    problem::Elicitation;
    report=nothing,
    validation=nothing,
    selections=(),
    history=(),
    metadata=(;),
    callbacks=(;),
)
    document = merge(
        Dict(
            "format" => "Credence session",
            "version" => 1,
            "environment" => _session_environment(),
            "history" => _session_pack(history, callbacks),
            "metadata" => _session_pack(metadata, callbacks),
        ),
        _session_state(problem, report, validation, selections, callbacks),
    )
    document["digest"]=_session_digest(document)
    mktemp(dirname(abspath(path))) do temporary, io
        TOML.print(io, document; sorted=true)
        close(io)
        mv(temporary, path; force=true)
    end
    path
end

function save_session(path, session::NamedTuple; callbacks=(;))
    save_session(
        path,
        session.problem;
        report=session.report,
        validation=session.validation,
        selections=session.selections,
        history=session.history,
        metadata=session.metadata,
        callbacks,
    )
end

"""
    load_session(path; callbacks=(;))

Restore a saved session as a named record with `problem`, `report`, `validation`,
`selections`, `history`, `metadata`, and `environment`. Measures are rebuilt from saved
parameters without optimization. Solver provenance retains its return code and type,
while callback implementations come from the supplied registry. Package versions and
source fingerprints must match before saved validation or acceptance is restored.
"""
function load_session(path; callbacks=(;))
    document = TOML.parsefile(path)
    get(document, "format", nothing) == "Credence session" &&
    get(document, "version", nothing) == 1 ||
        throw(ArgumentError("unsupported Credence session format"))
    digest=pop!(document, "digest", nothing)
    digest==_session_digest(document) || throw(
        ArgumentError(
            "saved session data changed; validation and selection cannot be restored",
        ),
    )
    environment=document["environment"]
    current=_session_environment()
    for name in ("Credence", "MeasureBase", "Distributions", "source_fingerprint")
        get(environment, name, nothing)==get(current, name, nothing) || throw(
            ArgumentError(
                "saved $name provenance differs from this environment; restore the original environment before loading",
            ),
        )
    end
    state=_session_restore_state(document, callbacks)
    (;
        state...,
        history=_session_unpack(document["history"], callbacks),
        metadata=_session_unpack(document["metadata"], callbacks),
        environment=document["environment"],
    )
end

_session_digest(document) =
    bytes2hex(SHA.sha256(sprint(io -> TOML.print(io, document; sorted=true))))

function _session_environment()
    files=String[]
    for directory in (@__DIR__, joinpath(dirname(@__DIR__), "ext"))
        append!(files, filter(path -> endswith(path, ".jl"), readdir(directory; join=true)))
    end
    contents=join(
        (
            replace(relpath(path, dirname(@__DIR__)), '\\'=>'/')*"\n"*replace(
                read(path, String),
                "\r\n"=>"\n",
            ) for path in sort(files)
        ),
        "\n",
    )
    environment=Dict(
        "julia"=>string(VERSION),
        "Credence"=>string(Base.pkgversion(Credence)),
        "MeasureBase"=>string(Base.pkgversion(MeasureBase)),
        "source_fingerprint"=>bytes2hex(SHA.sha256(contents)),
    )
    extension=Base.get_extension(Credence, :CredenceDistributionsExt)
    if extension!==nothing
        environment["Distributions"]=string(Base.pkgversion(extension.Distributions))
    end
    environment
end

function _session_state(problem, report, validation, selections, callbacks)
    report === nothing ||
        report.problem === problem ||
        throw(ArgumentError("the report belongs to another elicitation problem"))
    validation === nothing ||
        (report !== nothing && validation.report === report) ||
        throw(ArgumentError("validation must belong to the saved report"))
    Dict(
        "problem"=>_session_pack(problem, callbacks),
        "report"=>_session_pack_report(report, callbacks),
        "validation"=>_session_pack_validation(validation, callbacks),
        "selections"=>[
            _session_selection(s, report, validation, callbacks) for s in selections
        ],
    )
end
function _session_restore_state(data, callbacks)
    problem=_session_unpack(data["problem"], callbacks)
    report=_session_unpack_report(data["report"], problem, callbacks)
    validation=_session_unpack_validation(data["validation"], report, callbacks)
    selections=Tuple(
        _session_selection(s, report, validation, callbacks, Val(:load)) for
        s in data["selections"]
    )
    (; problem, report, validation, selections)
end

_session_tag(tag; fields...) =
    Dict{String,Any}("tag" => tag, (string(k) => v for (k, v) in pairs(fields))...)
_session_pack(x::Nothing, callbacks) = _session_tag("nothing")
_session_pack(x::Missing, callbacks) = _session_tag("missing")
_session_pack(x::Bool, callbacks) = _session_tag("bool"; value=x)
_session_pack(x::String, callbacks) = _session_tag("string"; value=x)
_session_pack(x::Symbol, callbacks) = _session_tag("symbol"; value=string(x))
_session_pack(x::Union{Integer,AbstractFloat}, callbacks) =
    _session_tag("number"; type=string(typeof(x)), value=string(x))
_session_pack(x::BigFloat, callbacks) =
    _session_tag("number"; type="BigFloat", value=string(x), precision=precision(x))
_session_pack(x::Rational, callbacks) = _session_tag(
    "rational";
    numerator=_session_pack(numerator(x), callbacks),
    denominator=_session_pack(denominator(x), callbacks),
)
_session_pack(x::Base.RefValue, callbacks) =
    _session_tag("reference_value"; value=_session_pack(x[], callbacks))
_session_pack(x::Tuple, callbacks) =
    _session_tag("tuple"; values=[_session_pack(v, callbacks) for v in x])
function _session_pack(x::NamedTuple, callbacks)
    if haskey(x, :problem) && haskey(x, :report) && x.problem isa Elicitation
        names=Tuple(
            k for k in keys(x) if !(k in (:problem, :report, :validation, :selections))
        )
        extra=NamedTuple{names}(Tuple(getproperty(x, k) for k in names))
        return _session_tag(
            "snapshot";
            names=string.(collect(keys(x))),
            state=_session_state(
                x.problem,
                x.report,
                get(x, :validation, nothing),
                get(x, :selections, ()),
                callbacks,
            ),
            extra=_session_pack(extra, callbacks),
        )
    end
    _session_tag(
        "record";
        names=string.(collect(keys(x))),
        values=[_session_pack(v, callbacks) for v in values(x)],
    )
end
_session_pack(x::Pair, callbacks) = _session_tag(
    "pair";
    first=_session_pack(x.first, callbacks),
    last=_session_pack(x.second, callbacks),
)
_session_pack(x::AbstractDict, callbacks) =
    _session_tag("dictionary"; values=[_session_pack(pair, callbacks) for pair in pairs(x)])
_session_pack(x::Colon, callbacks) = _session_tag("colon")
_session_pack(x::AbstractRange, callbacks) = _session_tag(
    "range";
    first=_session_pack(first(x), callbacks),
    step=_session_pack(step(x), callbacks),
    last=_session_pack(last(x), callbacks),
)
function _session_pack(x::AbstractArray, callbacks)
    axes(x) == map(Base.OneTo, size(x)) ||
        throw(ArgumentError("saved arrays need one-based axes"))
    _session_tag(
        "array";
        dimensions=collect(size(x)),
        element=_session_type(eltype(x)),
        values=[_session_pack(v, callbacks) for v in vec(x)],
    )
end
function _session_pack(x::Union{UpperTriangular,LowerTriangular,Symmetric}, callbacks)
    uplo=""
    if x isa Symmetric
        uplo=string(x.uplo)
    end
    _session_tag(string(nameof(typeof(x))); value=_session_pack(parent(x), callbacks), uplo)
end

const _session_numbers = (
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    BigInt,
    Float16,
    Float32,
    Float64,
    BigFloat,
)
function _session_type(T)
    T in (_session_numbers..., Bool, String, Symbol, Nothing, Missing, Any) &&
        return string(T)
    T <: NamedTuple && return Dict("record"=>true)
    if T isa Union
        return Dict("union" => [_session_type(t) for t in Base.uniontypes(T)])
    end
    throw(ArgumentError("array element type $T has no portable session representation"))
end
function _session_type(data::AbstractString)
    index = findfirst(
        T -> string(T) == data,
        (_session_numbers..., Bool, String, Symbol, Nothing, Missing, Any),
    )
    index === nothing &&
        throw(ArgumentError("unsupported saved numeric or array type $data"))
    (_session_numbers..., Bool, String, Symbol, Nothing, Missing, Any)[index]
end
function _session_type(data::AbstractDict)
    haskey(data, "record") && return NamedTuple
    Union{map(_session_type, data["union"])...}
end

_session_pack(x::VarName, callbacks) = _session_tag(
    "reference";
    name=string(getsym(x)),
    optic=_session_pack(getoptic(x), callbacks),
)
_session_pack(x::VarNames.Iden, callbacks) = _session_tag("identity")
_session_pack(x::VarNames.Property{name}, callbacks) where {name} = _session_tag(
    "property";
    name=string(name),
    child=_session_pack(VarNames.otail(x), callbacks),
)
_session_pack(x::VarNames.Index, callbacks) = _session_tag(
    "index";
    indices=_session_pack(x.ix, callbacks),
    keywords=_session_pack(x.kw, callbacks),
    child=_session_pack(VarNames.otail(x), callbacks),
)

const _session_structs = (
    :Quantile,
    :Mean,
    :Std,
    :Covariance,
    :Correlation,
    :CovarianceMatrix,
    :CorrelationMatrix,
    :Probability,
    :IntervalEvent,
    :Statistic,
    :PredictiveSimulator,
    :AtLeast,
    :AtMost,
    :Between,
    :EqualTo,
    :Joint,
    :Derived,
    :Predictive,
    :On,
)
const _session_distributions = (
    :Normal,
    :LogNormal,
    :Beta,
    :Gamma,
    :Weibull,
    :Exponential,
    :TDist,
    :Cauchy,
    :Laplace,
    :Dirichlet,
    :MvTDist,
    :MvNormal,
    :MvLogNormal,
)
function _session_pack(x::Type, callbacks)
    name=nameof(x)
    extension=Base.get_extension(Credence, :CredenceDistributionsExt)
    name in _session_distributions &&
    extension!==nothing &&
    getfield(extension.Distributions, name)===x ||
        throw(ArgumentError("constructor $x has no portable session representation"))
    _session_tag("distribution_constructor"; name=string(name))
end
function _session_pack(x, callbacks)
    name = nameof(typeof(x))
    if name in _session_structs && parentmodule(typeof(x))===Credence
        return _session_tag(
            "object";
            name=string(name),
            fields=[
                _session_pack(getfield(x, i), callbacks) for i in 1:fieldcount(typeof(x))
            ],
        )
    elseif x isa AbstractFamily
        return _session_family(x, callbacks)
    end
    for (id, entry) in pairs(callbacks)
        callback, version = _session_callback_entry(entry)
        if x === callback
            return _session_tag(
                "callback";
                id=string(id),
                fingerprint=_session_callback_fingerprint(callback, version, callbacks),
            )
        end
    end
    throw(
        ArgumentError("$(typeof(x)) needs an explicit callback registry entry for saving"),
    )
end
_session_pack(x::Hard, callbacks) = _session_tag(
    "hard";
    id=_session_pack(x.id, callbacks),
    statistic=_session_pack(x.statistic, callbacks),
    relation=_session_pack(x.relation, callbacks),
)
_session_pack(x::Approximate, callbacks) = _session_tag(
    "approximate";
    id=_session_pack(x.id, callbacks),
    statistic=_session_pack(x.statistic, callbacks),
    target=_session_pack(x.target, callbacks),
    scale=_session_pack(x.scale, callbacks),
    importance=_session_pack(x.importance, callbacks),
)
_session_pack(x::MonteCarlo, callbacks) =
    _session_tag("montecarlo"; n=x.n, alpha=_session_pack(x.alpha, callbacks))
_session_pack(x::Analytic, callbacks) = _session_tag("analytic")
_session_pack(x::Estimate, callbacks) = _session_tag(
    "estimate";
    value=_session_pack(x.value, callbacks),
    lower=_session_pack(x.lower, callbacks),
    upper=_session_pack(x.upper, callbacks),
    evidence=string(x.evidence),
    n=x.n,
)
_session_pack(x::Elicitation, callbacks) = _session_tag(
    "problem";
    beliefs=_session_pack(x.beliefs, callbacks),
    candidates=_session_pack(x.candidates, callbacks),
)

function _session_family(x, callbacks)
    parentmodule(typeof(x))===Credence || throw(
        ArgumentError("custom family definitions need a registered candidate builder"),
    )
    fields = NamedTuple{fieldnames(typeof(x))}(
        Tuple(getfield(x, i) for i in 1:fieldcount(typeof(x))),
    )
    kind = nothing
    if nameof(typeof(x)) == :ScalarFamily
        kind = typeof(x).parameters[1]
    end
    _session_tag(
        "family";
        name=string(nameof(typeof(x))),
        kind=_session_pack(kind, callbacks),
        fields=_session_pack(fields, callbacks),
    )
end
function _session_pack(x::Candidate, callbacks)
    builder = x.build
    builder isa NamedBuilder && (builder=builder.f)
    definition = if x.family !== nothing
        _session_pack(x.family, callbacks)
    elseif builder isa IndependentBuilder
        _session_tag("independent"; parts=_session_pack(builder.parts, callbacks))
    else
        _session_pack(builder, callbacks)
    end
    _session_tag(
        "candidate";
        id=_session_pack(x.id, callbacks),
        definition,
        starts=_session_pack(x.starts, callbacks),
        lower=_session_pack(x.lower, callbacks),
        upper=_session_pack(x.upper, callbacks),
        scope=_session_pack(x.scope, callbacks),
        requirements=_session_pack(x.requirements, callbacks),
    )
end

struct SavedSolver
    retcode::String
    type::String
end
struct SavedError <: Exception
    message::String
    type::String
end
Base.showerror(io::IO, x::SavedError) = print(io, x.type, ": ", x.message)
_session_error(::Nothing) = _session_tag("nothing")
_session_error(x) =
    _session_tag("error"; message=sprint(showerror, x), type=string(typeof(x)))
_session_error(x::SavedError) = _session_tag("error"; message=x.message, type=x.type)
_session_solver(::Nothing) = _session_tag("nothing")
_session_solver(x) =
    _session_tag("solver"; retcode=string(x.retcode), type=string(typeof(x)))
_session_solver(x::SavedSolver) = _session_tag("solver"; retcode=x.retcode, type=x.type)
_session_pack_report(::Nothing, callbacks) = _session_tag("nothing")
function _session_pack_report(report, callbacks)
    results = map(report.results) do result
        index = findfirst(
            c -> _resultid(c, 1) == _resultid(result.candidate, 1),
            report.problem.candidates,
        )
        index === nothing &&
            throw(ArgumentError("result candidate is absent from its report"))
        Dict(
            "candidate" => index,
            "id" => _session_pack(result.id, callbacks),
            "parameters" => _session_pack(result.parameters, callbacks),
            "solver" => _session_solver(result.solver),
            "error" => _session_error(result.error),
        )
    end
    Dict(
        "tag" => "report",
        "results" => collect(results),
        "sampling" => _session_pack(report.sampling, callbacks),
        "seed" => _session_pack(report.seed, callbacks),
    )
end
_session_pack_validation(::Nothing, callbacks) = _session_tag("nothing")
function _session_pack_validation(validation, callbacks)
    checks = [
        [
            Dict(
                "id"=>_session_pack(c.id, callbacks),
                "estimate"=>_session_pack(c.estimate, callbacks),
                "status"=>string(c.status),
                "error"=>_session_error(c.error),
            ) for c in row
        ] for row in validation.checks
    ]
    Dict(
        "tag"=>"validation",
        "checks"=>checks,
        "plan"=>_session_pack(validation.plan, callbacks),
        "seed"=>_session_pack(validation.seed, callbacks),
    )
end
function _session_selection(selection, report, validation, callbacks)
    _session_verify_selection(selection)
    selection.report === report ||
        throw(ArgumentError("selection must belong to the saved report"))
    shared=selection.validation === validation
    retained=_session_tag("nothing")
    if !shared
        retained=_session_pack_validation(selection.validation, callbacks)
    end
    Dict(
        "index"=>selection.index,
        "role"=>string(selection.role),
        "rationale"=>_session_pack(selection.rationale, callbacks),
        "shared_validation"=>shared,
        "validation"=>retained,
    )
end
function _session_pack(selection::Selection, callbacks)
    _session_verify_selection(selection)
    _session_tag(
        "selection";
        problem=_session_pack(selection.report.problem, callbacks),
        report=_session_pack_report(selection.report, callbacks),
        validation=_session_pack_validation(selection.validation, callbacks),
        index=selection.index,
        role=string(selection.role),
        rationale=_session_pack(selection.rationale, callbacks),
    )
end

function _session_verify_selection(selection)
    validation=selection.validation
    validation===nothing ||
        validation.report===selection.report ||
        throw(ArgumentError("selection validation belongs to another report"))
end

function _session_unpack(data, callbacks)
    tag=data["tag"]
    tag == "nothing" && return nothing
    tag == "missing" && return missing
    tag == "bool" && return data["value"]
    tag == "string" && return data["value"]
    tag == "symbol" && return Symbol(data["value"])
    tag == "reference_value" && return Ref(_session_unpack(data["value"], callbacks))
    if tag == "number"
        if data["type"]=="BigFloat"
            return setprecision(BigFloat, data["precision"]) do
                parse(BigFloat, data["value"])
            end
        end
        return parse(_session_type(data["type"]), data["value"])
    end
    tag == "rational" &&
        return _session_unpack(data["numerator"], callbacks)//_session_unpack(
            data["denominator"],
            callbacks,
        )
    tag == "tuple" && return Tuple(_session_unpack(v, callbacks) for v in data["values"])
    tag == "record" && return NamedTuple{Tuple(Symbol.(data["names"]))}(
        Tuple(_session_unpack(v, callbacks) for v in data["values"]),
    )
    if tag=="snapshot"
        values=merge(
            _session_restore_state(data["state"], callbacks),
            _session_unpack(data["extra"], callbacks),
        )
        names=Tuple(Symbol.(data["names"]))
        return NamedTuple{names}(Tuple(getproperty(values, k) for k in names))
    end
    tag == "pair" && return _session_unpack(data["first"], callbacks)=>_session_unpack(
        data["last"],
        callbacks,
    )
    tag == "dictionary" &&
        return Dict(_session_unpack(value, callbacks) for value in data["values"])
    tag == "colon" && return Colon()
    tag == "range" && return _session_unpack(data["first"], callbacks):_session_unpack(
        data["step"],
        callbacks,
    ):_session_unpack(data["last"], callbacks)
    if tag == "array"
        T=_session_type(data["element"])
        return reshape(
            T[_session_unpack(v, callbacks) for v in data["values"]],
            Tuple(data["dimensions"]),
        )
    elseif tag in ("UpperTriangular", "LowerTriangular", "Symmetric")
        value=_session_unpack(data["value"], callbacks)
        tag == "UpperTriangular" && return UpperTriangular(value)
        tag == "LowerTriangular" && return LowerTriangular(value)
        return Symmetric(value, Symbol(data["uplo"]))
    elseif tag == "reference"
        return VarName{Symbol(data["name"])}(_session_unpack(data["optic"], callbacks))
    elseif tag == "identity"
        return VarNames.Iden()
    elseif tag == "property"
        return VarNames.Property{Symbol(data["name"])}(
            _session_unpack(data["child"], callbacks),
        )
    elseif tag == "index"
        return VarNames.Index(
            _session_unpack(data["indices"], callbacks),
            _session_unpack(data["keywords"], callbacks),
            _session_unpack(data["child"], callbacks),
        )
    elseif tag == "callback"
        matches=[entry for (id, entry) in pairs(callbacks) if string(id)==data["id"]]
        length(matches)==1 ||
            throw(ArgumentError("callback $(data["id"]) must be supplied exactly once"))
        callback, version = _session_callback_entry(only(matches))
        _session_callback_fingerprint(callback, version, callbacks)==data["fingerprint"] ||
            throw(
                ArgumentError("callback $(data["id"]) changed since the session was saved"),
            )
        return callback
    elseif tag == "object"
        name=Symbol(data["name"])
        name in _session_structs || throw(ArgumentError("unsupported saved object $name"))
        fields=Tuple(_session_unpack(v, callbacks) for v in data["fields"])
        name==:Joint && return Joint(only(fields)...)
        return getfield(Credence, name)(fields...)
    elseif tag=="distribution_constructor"
        name=Symbol(data["name"])
        name in _session_distributions ||
            throw(ArgumentError("unsupported saved distribution constructor $name"))
        extension=Base.get_extension(Credence, :CredenceDistributionsExt)
        extension===nothing &&
            throw(ArgumentError("load Distributions to restore its family constructors"))
        return getfield(extension.Distributions, name)
    elseif tag == "hard"
        return Hard(
            _session_unpack(data["id"], callbacks),
            _session_unpack(data["statistic"], callbacks),
            _session_unpack(data["relation"], callbacks),
        )
    elseif tag == "approximate"
        return Approximate(
            _session_unpack(data["id"], callbacks),
            _session_unpack(data["statistic"], callbacks),
            _session_unpack(data["target"], callbacks);
            scale=_session_unpack(data["scale"], callbacks),
            importance=_session_unpack(data["importance"], callbacks),
        )
    elseif tag == "montecarlo"
        return MonteCarlo(data["n"]; alpha=_session_unpack(data["alpha"], callbacks))
    elseif tag == "analytic"
        return Analytic()
    elseif tag == "estimate"
        return Estimate(
            _session_unpack(data["value"], callbacks);
            lower=_session_unpack(data["lower"], callbacks),
            upper=_session_unpack(data["upper"], callbacks),
            evidence=Symbol(data["evidence"]),
            n=data["n"],
        )
    elseif tag == "problem"
        return Elicitation(
            _session_unpack(data["beliefs"], callbacks),
            _session_unpack(data["candidates"], callbacks),
        )
    elseif tag == "family"
        return _session_unpack_family(data, callbacks)
    elseif tag == "candidate"
        family=nothing
        definition=data["definition"]
        id=_session_unpack(data["id"], callbacks)
        if definition["tag"]=="family"
            family=_session_unpack(definition, callbacks)
            builder=_family_candidate(id, family).build
            builder isa NamedBuilder && (builder=builder.f)
        elseif definition["tag"]=="independent"
            builder=IndependentBuilder(_session_unpack(definition["parts"], callbacks))
        else
            builder=_session_unpack(definition, callbacks)
        end
        return Candidate(
            id,
            builder,
            _session_unpack(data["starts"], callbacks);
            lower=_session_unpack(data["lower"], callbacks),
            upper=_session_unpack(data["upper"], callbacks),
            scope=_session_unpack(data["scope"], callbacks),
            requirements=_session_unpack(data["requirements"], callbacks),
            family,
        )
    elseif tag == "selection"
        problem=_session_unpack(data["problem"], callbacks)
        report=_session_unpack_report(data["report"], problem, callbacks)
        validation=_session_unpack_validation(data["validation"], report, callbacks)
        return select(
            report,
            report.results[data["index"]].id;
            validation,
            role=Symbol(data["role"]),
            rationale=_session_unpack(data["rationale"], callbacks),
        )
    end
    throw(ArgumentError("unsupported session value tag $tag"))
end

function _session_unpack_family(data, callbacks)
    name=Symbol(data["name"])
    fields=_session_unpack(data["fields"], callbacks)
    if name==:ScalarFamily
        kind=_session_unpack(data["kind"], callbacks)
        kinds=(
            :gamma,
            :weibull,
            :exponential,
            :student_t,
            :cauchy,
            :laplace,
            :half_normal,
            :half_student_t,
        )
        constructors=(
            :GammaFamily,
            :WeibullFamily,
            :ExponentialFamily,
            :StudentTFamily,
            :CauchyFamily,
            :LaplaceFamily,
            :HalfNormalFamily,
            :HalfStudentTFamily,
        )
        index=findfirst(==(kind), kinds)
        index===nothing && throw(ArgumentError("unsupported saved family kind $kind"))
        return getfield(Credence, constructors[index])(;
            fields.parameters...,
            starts=fields.starts,
        )
    elseif name==:ScaledBetaFamily
        return ScaledBetaFamily(fields.lower, fields.upper; starts=fields.starts)
    elseif name==:TruncatedFamily
        return TruncatedFamily(
            fields.base,
            fields.lower,
            fields.upper;
            starts=fields.starts,
        )
    elseif name==:IIDFamily
        return IIDFamily(fields.family, fields.dimensions)
    elseif name in
           (:MvNormalFamily, :MvLogNormalFamily, :DirichletFamily, :MvStudentTFamily)
        rest=Base.structdiff(fields, (dimension=fields.dimension,))
        return getfield(Credence, name)(fields.dimension; rest...)
    elseif name in (:NormalFamily, :LogNormalFamily)
        return getfield(Credence, name)(; fields...)
    end
    throw(ArgumentError("unsupported saved family $name"))
end
function _session_unpack_report(data, problem, callbacks)
    data["tag"]=="nothing" && return nothing
    results=map(data["results"]) do row
        candidate=problem.candidates[row["candidate"]]
        id=_session_unpack(row["id"], callbacks)
        parameters=_session_unpack(row["parameters"], callbacks)
        solverdata=row["solver"]
        solver=nothing
        solverdata["tag"]=="nothing" ||
            (solver=SavedSolver(solverdata["retcode"], solverdata["type"]))
        error=_session_unpack_error(row["error"])
        if error===nothing
            values=_flatten(parameters)
            return _result(candidate, last(id), values, candidate.build(values), solver)
        end
        FitResult(id, candidate, parameters, nothing, solver, error)
    end
    FitReport(
        problem,
        Tuple(results),
        _session_unpack(data["sampling"], callbacks),
        _session_unpack(data["seed"], callbacks),
    )
end
function _session_unpack_error(data)
    data["tag"]=="nothing" && return nothing
    SavedError(data["message"], data["type"])
end
function _session_unpack_validation(data, report, callbacks)
    data["tag"]=="nothing" && return nothing
    report===nothing && throw(ArgumentError("saved validation has no report"))
    checks=Tuple(
        Tuple(
            Check(
                _session_unpack(c["id"], callbacks),
                _session_unpack(c["estimate"], callbacks),
                Symbol(c["status"]),
                _session_unpack_error(c["error"]),
            ) for c in row
        ) for row in data["checks"]
    )
    Validation(
        report,
        checks,
        _session_unpack(data["plan"], callbacks),
        _session_unpack(data["seed"], callbacks),
    )
end
function _session_selection(data, report, validation, callbacks, ::Val{:load})
    localvalidation=validation
    data["shared_validation"] ||
        (localvalidation=_session_unpack_validation(data["validation"], report, callbacks))
    select(
        report,
        report.results[data["index"]].id;
        validation=localvalidation,
        role=Symbol(data["role"]),
        rationale=_session_unpack(data["rationale"], callbacks),
    )
end

function _session_callback_entry(entry)
    if entry isa NamedTuple && haskey(entry, :callback)
        haskey(entry, :version) ||
            throw(ArgumentError("a callback descriptor needs a version"))
        return entry.callback, string(entry.version)
    end
    entry, nothing
end
function _session_callback_fingerprint(callback, version, callbacks)
    recipes=map(collect(methods(callback))) do method
        file=Base.find_source_file(String(method.file))
        if file!==nothing && isfile(file)
            return Dict(
                "source"=>bytes2hex(SHA.sha256(replace(read(file, String), "\r\n"=>"\n"))),
                "line"=>method.line,
                "arguments"=>method.nargs,
            )
        end
        version===nothing &&
            throw(ArgumentError("REPL callbacks need an explicit reconstruction version"))
        Dict("lowered"=>repr(Base.uncompressed_ast(method).code), "arguments"=>method.nargs)
    end
    captures=map(1:fieldcount(typeof(callback))) do i
        value=getfield(callback, i)
        try
            _session_pack(value, callbacks)
        catch error
            error isa ArgumentError || rethrow()
            version===nothing && throw(
                ArgumentError(
                    "opaque callback captures need an explicit reconstruction version",
                ),
            )
            Dict("opaque_capture"=>string(typeof(value)))
        end
    end
    payload=Dict(
        "methods"=>recipes,
        "captures"=>captures,
        "globals"=>_session_callback_globals(callback, version, callbacks),
        "version"=>something(version, ""),
    )
    bytes2hex(SHA.sha256(sprint(io -> TOML.print(io, payload; sorted=true))))
end

function _session_global_refs!(references, value)
    if value isa GlobalRef
        push!(references, value)
    elseif value isa Expr
        foreach(v -> _session_global_refs!(references, v), value.args)
    elseif value isa Core.ReturnNode && isdefined(value, :val)
        _session_global_refs!(references, value.val)
    elseif value isa Core.GotoIfNot
        _session_global_refs!(references, value.cond)
    end
    references
end
function _session_callback_globals(callback, version, callbacks)
    references=GlobalRef[]
    for method in methods(callback)
        code=Base.uncompressed_ast(method)
        code===nothing && continue
        foreach(value -> _session_global_refs!(references, value), code.code)
    end
    globals=Dict{String,Any}()
    trusted=(
        Base,
        Core,
        Credence,
        MeasureBase,
        Random,
        parentmodule(mean),
        LinearAlgebra,
        VarNames,
    )
    extension=Base.get_extension(Credence, :CredenceDistributionsExt)
    if extension!==nothing
        trusted=(trusted..., extension.Distributions)
    end
    for reference in unique(references)
        isdefined(reference.mod, reference.name) || continue
        value=getfield(reference.mod, reference.name)
        value===callback && continue
        if value isa Function || value isa Type || value isa Module
            owner=reference.mod
            if value isa Function
                owner=parentmodule(typeof(value))
            elseif value isa Type
                owner=parentmodule(value)
            else
                owner=value
            end
            while !(owner in trusted) && parentmodule(owner)!==owner
                owner=parentmodule(owner)
            end
            owner in trusted && continue
            version===nothing && throw(
                ArgumentError(
                    "callbacks using external helpers need an explicit reconstruction version",
                ),
            )
            continue
        end
        key=string(reference.mod)*"."*string(reference.name)
        try
            globals[key]=_session_pack(value, callbacks)
        catch error
            error isa ArgumentError || rethrow()
            version===nothing && throw(
                ArgumentError(
                    "callbacks using opaque global state need an explicit reconstruction version",
                ),
            )
            globals[key]=Dict("opaque_global"=>string(typeof(value)))
        end
    end
    globals
end
