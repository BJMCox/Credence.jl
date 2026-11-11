function Credence._compound_base(family::Credence.GroupHorseshoeFamily, parameters)
    scale = Credence._compound_parameter(family, parameters, Val(:scale))
    unit = one(float(scale))
    half_cauchy =
        asmeasure(truncated(Cauchy(zero(unit), unit), zero(unit), oftype(unit, Inf)))
    MeasureBase.productmeasure((
        z=MeasureBase.productmeasure(
            fill(asmeasure(Normal(zero(unit), unit)), length(family.groups)),
        ),
        global_scale=half_cauchy,
        group_scales=MeasureBase.productmeasure(fill(half_cauchy, maximum(family.indices))),
        local_scales=MeasureBase.productmeasure(fill(half_cauchy, length(family.groups))),
    ))
end
function Credence._compound_linear(family::Credence.GroupHorseshoeFamily, parameters, point)
    scale =
        Credence._compound_parameter(family, parameters, Val(:scale)) * point.global_scale
    Diagonal([
        scale * point.group_scales[group] * local_scale for
        (group, local_scale) in zip(family.indices, point.local_scales)
    ])
end
