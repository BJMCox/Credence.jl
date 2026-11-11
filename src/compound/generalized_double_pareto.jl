"""
    GeneralizedDoubleParetoFamily(dimension; shape=1.0, scale=nothing, starts=nothing)

Construct a generalized double Pareto prior with marginal density
`(1 + abs(x)/(shape*scale))^(-shape-1) / (2scale)` for each coefficient.
Here `scale` is the paper's `ξ` and `shape` is its `α`. The mean exists for
`shape > 1`, and the variance is `2(shape*scale)^2 / ((shape-1)*(shape-2))`
for `shape > 2`. Use quantiles or event probabilities for heavier tails.

Independent unit-scale Laplace draws divided by independent unit-rate Gamma
variables give the coefficient mixture. The joint record retains `coefficients`
and `local_rates`. Physical conditional Laplace rates are
`local_rates / (shape*scale)`. Numeric keywords fix hyperparameters; `nothing`
fits `log_shape` or `log_scale` through `starts`.
"""
struct GeneralizedDoubleParetoFamily{A,S,T} <: AbstractCompoundFamily
    dimension::Int
    shape::A
    scale::S
    starts::T
end
function GeneralizedDoubleParetoFamily(n::Integer; shape=1.0, scale=nothing, starts=nothing)
    n > 0 || throw(ArgumentError("generalized double Pareto dimension must be positive"))
    _compound_positive(shape, :shape)
    _compound_positive(scale, :scale)
    GeneralizedDoubleParetoFamily(n, shape, scale, starts)
end

_compound_shape(family::GeneralizedDoubleParetoFamily) = (family.dimension,)
function _compound_starts(family::GeneralizedDoubleParetoFamily)
    if family.starts isa NamedTuple
        return (family.starts,)
    elseif family.starts !== nothing
        return family.starts
    end
    initial = (;)
    if family.shape === nothing
        initial = (; initial..., log_shape=0.0)
    end
    if family.scale === nothing
        initial = (; initial..., log_scale=0.0)
    end
    (initial,)
end
