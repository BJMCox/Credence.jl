_flatppl_coefficients(family, parameters, reference, path) = throw(
    ArgumentError("no FlatPPL coefficient translation for $(typeof(family)) at $path"),
)

function _flatppl(prior::CompoundMeasure, path)
    transform = MeasureBase.gettransform(prior)
    parts = MeasureBase.marginals(MeasureBase.boundmeasure(prior))
    auxiliary = _flatppl_block(
        "joint(",
        (
            string(name) *
            " = " *
            _flatppl_compound_auxiliary(part, path * "." * string(name)) for
            (name, part) in pairs(parts)
        ),
        ")",
    )
    coefficients = _flatppl_coefficients(
        transform.family,
        transform.parameters,
        "r",
        path * ".coefficients",
    )
    _flatppl_block(
        "jointchain(",
        (auxiliary, "r -> joint(coefficients = " * coefficients * ")"),
        ")",
    )
end

# Iid arrays retain their shape and avoid expanding identical scalar factors.
_flatppl_compound_auxiliary(part, path) = _flatppl(part, path)
function _flatppl_compound_auxiliary(
    part::MeasureBase.ProductMeasure{<:AbstractArray},
    path,
)
    parts = MeasureBase.marginals(part)
    _flatppl_axes(parts, path)
    isempty(parts) && throw(ArgumentError("empty auxiliary prior at $path"))
    law = _flatppl(first(parts), path)
    if all(part -> _flatppl(part, path) == law, parts)
        shape = if ndims(parts) == 1
            string(length(parts))
        else
            _flatppl_value(collect(size(parts)), path)
        end
        return "iid(" * law * ", " * shape * ")"
    end
    _flatppl_product(parts, path)
end
