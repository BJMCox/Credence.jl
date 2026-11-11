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
    # The scale can be representable when the variance budget product is not.
    log_budget = (log(draw.r2) - log1p(-draw.r2) - log(2one(draw.r2))) / 2
    scales = exp.(
        log(family.noise_scale) .+ log_budget .+ log.(draw.allocation) ./ 2 .-
        log.(family.predictor_scales),
    )
    Diagonal(scales)
end
