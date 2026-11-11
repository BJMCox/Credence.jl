function Credence._compound_base(family::Credence.GeneralizedDoubleParetoFamily, parameters)
    shape = Credence._compound_parameter(family, parameters, Val(:shape))
    unit = one(shape)
    MeasureBase.productmeasure((
        z=MeasureBase.productmeasure(
            fill(asmeasure(Laplace(zero(unit), unit)), family.dimension),
        ),
        local_rates=MeasureBase.productmeasure(
            fill(asmeasure(Gamma(shape, unit)), family.dimension),
        ),
    ))
end

function Credence._compound_linear(
    family::Credence.GeneralizedDoubleParetoFamily,
    parameters,
    point,
)
    shape = Credence._compound_parameter(family, parameters, Val(:shape))
    scale = Credence._compound_parameter(family, parameters, Val(:scale))
    Diagonal((shape*scale) ./ point.local_rates)
end
