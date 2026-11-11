"""
    flatppl(prior)
    flatppl(selection)

Return a FlatPPL measure expression for a supported prior, preserving its names,
shape, dependence, and normalization. Unsupported measures raise an `ArgumentError`
that identifies the component requiring a translator.
"""
flatppl(prior::MeasureBase.AbstractMeasure) = _flatppl(prior, "prior")
flatppl(selection::Selection) = flatppl(measure(selection))

function _flatppl_block(open, parts, close)
    lines = ("    " * replace(part, "\n" => "\n    ") for part in parts)
    open * "\n" * join(lines, ",\n") * "\n" * close
end

_flatppl(prior, path) =
    throw(ArgumentError("no FlatPPL translation for $(typeof(prior)) at $path"))
_flatppl(::MeasureBase.StdNormal, path) = "Normal(0.0, 1.0)"
_flatppl(::MeasureBase.StdUniform, path) = "Uniform(interval(0.0, 1.0))"
_flatppl(::MeasureBase.StdExponential, path) = "Exponential(1.0)"
_flatppl(prior::MeasureBase.Dirac, path) =
    "Dirac(" * _flatppl_value(MeasureBase.atomof(prior), path) * ")"

function _flatppl_value(x::Union{Float16,Float32,Float64}, path)
    isnan(x) && throw(ArgumentError("NaN at $path has no FlatPPL prior translation"))
    if isinf(x)
        if x > 0
            return "inf"
        else
            return "-inf"
        end
    end
    string(Float64(x))
end
function _flatppl_value(x::Integer, path)
    abs(big(x)) <= 2^53 ||
        throw(ArgumentError("integer at $path exceeds exact FlatPPL numeric precision"))
    string(x)
end
_flatppl_value(x, path) = _flatppl(x, path)
function _flatppl_axes(x, path)
    axes(x) == map(Base.OneTo, size(x)) ||
        throw(ArgumentError("FlatPPL requires one-based array axes at $path"))
end
function _flatppl_value(x::AbstractVector, path)
    _flatppl_axes(x, path)
    "[" * join((_flatppl_value(v, "$path[$i]") for (i, v) in enumerate(x)), ", ") * "]"
end
function _flatppl_value(x::AbstractMatrix, path)
    _flatppl_axes(x, path)
    _flatppl_block(
        "rowstack([",
        (
            _flatppl_value(collect(row), "$path[$i, :]") for
            (i, row) in enumerate(eachrow(x))
        ),
        "])",
    )
end

function _flatppl(prior::MeasureBase.AbstractProductMeasure, path)
    _flatppl_product(MeasureBase.marginals(prior), path)
end
function _flatppl_product(parts::NamedTuple, path)
    fields = map(keys(parts)) do name
        occursin(r"^[A-Za-z][A-Za-z0-9_]*$", string(name)) &&
        !(string(name) in ("in", "true", "false", "all", "only")) ||
            throw(ArgumentError("field $name at $path is not a FlatPPL identifier"))
        string(name) * " = " * _flatppl(getproperty(parts, name), "$path.$name")
    end
    _flatppl_block("joint(", fields, ")")
end
function _flatppl_product(parts::AbstractVector, path)
    _flatppl_axes(parts, path)
    all(part -> MeasureBase.mspace_elsize(part) == (), parts) ||
        throw(ArgumentError("FlatPPL vector products require scalar factors at $path"))
    _flatppl_block(
        "joint(",
        (_flatppl(part, "$path[$i]") for (i, part) in enumerate(parts)),
        ")",
    )
end
_flatppl_product(parts, path) = _flatppl(parts, path)
function _flatppl(prior::MeasureBase.PowerMeasure, path)
    base = MeasureBase.pwr_base(prior)
    MeasureBase.mspace_elsize(base) == () ||
        throw(ArgumentError("FlatPPL iid export requires scalar factors at $path"))
    dimensions = size(prior)
    MeasureBase.pwr_axes(prior) == map(Base.OneTo, dimensions) ||
        throw(ArgumentError("FlatPPL requires one-based array axes at $path"))
    shape = if length(dimensions) == 1
        string(only(dimensions))
    else
        _flatppl_value(collect(dimensions), path)
    end
    "iid(" * _flatppl(base, path * "[]") * ", " * shape * ")"
end
