function Credence._compound_base(family::Credence.FunctionalHorseshoeFamily, parameters)
    scale = Credence._compound_parameter(family, parameters, Val(:scale))
    MeasureBase.productmeasure((
        z=MeasureBase.productmeasure(
            fill(asmeasure(Normal(zero(scale), one(scale))), size(family.design, 2)),
        ),
        shrinkage=asmeasure(Beta(family.a, family.b)),
    ))
end

function Credence._compound_linear(
    family::Credence.FunctionalHorseshoeFamily,
    parameters,
    x,
)
    scale = Credence._compound_parameter(family, parameters, Val(:scale))
    deviation = scale * sqrt((one(x.shrinkage) - x.shrinkage) / x.shrinkage)
    scales = vcat(
        fill(family.baseline_scale, family.baseline_dimension),
        fill(deviation, size(family.design, 2) - family.baseline_dimension),
    )
    Credence.FunctionalFactor(family, scales)
end
