function Credence._compound_base(family::Credence.NormalGammaFamily, parameters)
    shape = Credence._compound_parameter(family, parameters, Val(:shape))
    scale = Credence._compound_parameter(family, parameters, Val(:scale))
    unit = one(scale)
    MeasureBase.productmeasure((
        z=MeasureBase.productmeasure(
            fill(asmeasure(Normal(zero(unit), unit)), family.dimension),
        ),
        local_variances=MeasureBase.productmeasure(
            fill(asmeasure(Gamma(shape, 2scale^2)), family.dimension),
        ),
    ))
end
Credence._compound_linear(family::Credence.NormalGammaFamily, parameters, point) =
    Diagonal(sqrt.(point.local_variances))
