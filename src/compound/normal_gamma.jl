"""
    NormalGammaFamily(dimension; shape=nothing, scale=nothing, starts=nothing)

Construct the Normal–Gamma variance-mixture prior of Griffin and Brown (2010).
Each coefficient has conditional law `Normal(0, sqrt(local_variance))`, with
independent `local_variance ~ Gamma(shape, 2 * scale^2)` in shape-scale form.
The marginal variance is `2 * shape * scale^2` and excess kurtosis is `3 / shape`.
At `shape=1`, the coefficient marginal is `Laplace(0, scale)`.

Numeric keyword values fix the parameters. Set either to `nothing` to fit it
through `log_shape` or `log_scale`. Draws retain `coefficients` and
`local_variances` in one joint record.
"""
struct NormalGammaFamily{A,S,T} <: AbstractCompoundFamily
    dimension::Int
    shape::A
    scale::S
    starts::T
end
function NormalGammaFamily(n::Integer; shape=nothing, scale=nothing, starts=nothing)
    n > 0 || throw(ArgumentError("Normal–Gamma dimension must be positive"))
    _compound_positive(shape, :shape)
    _compound_positive(scale, :scale)
    NormalGammaFamily(n, shape, scale, starts)
end

_compound_shape(family::NormalGammaFamily) = (family.dimension,)
function _compound_starts(family::NormalGammaFamily)
    initial = NamedTuple()
    if family.shape === nothing
        initial = (; initial..., log_shape=0.0)
    end
    if family.scale === nothing
        initial = (; initial..., log_scale=0.0)
    end
    if family.starts === nothing
        return (initial,)
    elseif family.starts isa NamedTuple
        return (family.starts,)
    end
    family.starts
end
