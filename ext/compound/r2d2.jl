function Credence._compound_base(family::Credence.R2D2Family, parameters)
    mu = Credence._r2d2_mean(family, parameters)
    precision = Credence._compound_parameter(family, parameters, Val(:precision_r2))
    concentration = Credence._compound_parameter(family, parameters, Val(:concentration))
    MeasureBase.productmeasure((
        z=MeasureBase.productmeasure(
            fill(asmeasure(Laplace(zero(mu), one(mu))), family.dimension),
        ),
        r2=asmeasure(Beta(mu * precision, (one(mu)-mu) * precision)),
        allocation=asmeasure(Dirichlet(fill(concentration, family.dimension))),
    ))
end

function Credence._compound_linear(family::Credence.R2D2Family, parameters, draw)
    budget = draw.r2 / (one(draw.r2)-draw.r2)
    scales =
        family.noise_scale .* sqrt.(draw.allocation .* (budget/2)) ./
        family.predictor_scales
    Diagonal(scales)
end
