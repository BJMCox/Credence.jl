"""
    R2D2Family(dimension; noise_scale, predictor_scales, mean_r2=nothing,
               precision_r2=nothing, concentration=1/dimension, starts=nothing)

Fit the marginal R2-D2 prior for Gaussian linear regression. Predictors are centered,
`predictor_scales` gives their standard deviations, and `noise_scale` is the residual
standard deviation. Both must be supplied in the model's units.

The variance budget `r2` has a Beta law with shapes `mean_r2 * precision_r2` and
`(1 - mean_r2) * precision_r2`. A symmetric Dirichlet with per-component
`concentration` allocates that budget. Conditional coefficients are Laplace with scales
`noise_scale * sqrt.(allocation * r2 / (2 * (1-r2))) ./ predictor_scales`.
Here `r2` describes the prior variance budget after averaging over coefficients and
predictors. It differs from a realized coefficient vector's explained variance.

Fixed keyword values remove fitting coordinates. Free `mean_r2` uses `logit_mean_r2`;
free positive parameters use `log_precision_r2` and `log_concentration`. The measure
returns a record with `coefficients`, `r2`, and `allocation`.
"""
struct R2D2Family{M,P,C,N,X,S} <: AbstractCompoundFamily
    dimension::Int
    mean_r2::M
    precision_r2::P
    concentration::C
    noise_scale::N
    predictor_scales::X
    starts::S
end

function R2D2Family(
    dimension::Integer;
    noise_scale,
    predictor_scales,
    mean_r2=nothing,
    precision_r2=nothing,
    concentration=inv(float(dimension)),
    starts=nothing,
)
    dimension > 0 || throw(ArgumentError("R2D2 dimension must be positive"))
    mean_r2 === nothing ||
        (isfinite(mean_r2) && 0 < mean_r2 < 1) ||
        throw(ArgumentError("mean_r2 must lie in (0,1)"))
    _compound_positive(precision_r2, :precision_r2)
    _compound_positive(concentration, :concentration)
    _compound_positive(noise_scale, :noise_scale)
    noise_scale === nothing && throw(ArgumentError("noise_scale must be fixed"))
    predictor_scales isa AbstractVector && length(predictor_scales) == dimension ||
        throw(DimensionMismatch("predictor_scales must have one entry per coefficient"))
    all(x -> isfinite(x) && x > 0, predictor_scales) ||
        throw(ArgumentError("predictor_scales must be positive and finite"))
    R2D2Family(
        dimension,
        mean_r2,
        precision_r2,
        concentration,
        noise_scale,
        copy(predictor_scales),
        starts,
    )
end

_compound_shape(family::R2D2Family) = (family.dimension,)
function _compound_starts(family::R2D2Family)
    if family.starts isa NamedTuple
        return (family.starts,)
    elseif family.starts !== nothing
        return family.starts
    end
    initial = (;)
    if family.mean_r2 === nothing
        initial = (; initial..., logit_mean_r2=0.0)
    end
    if family.precision_r2 === nothing
        initial = (; initial..., log_precision_r2=log(2.0))
    end
    if family.concentration === nothing
        initial = (; initial..., log_concentration=(-log(float(family.dimension))))
    end
    (initial,)
end

function _r2d2_mean(family, parameters)
    family.mean_r2 === nothing || return family.mean_r2
    x = parameters.logit_mean_r2
    if x >= 0
        return inv(one(x) + exp(-x))
    end
    e = exp(x)
    e / (one(e) + e)
end
