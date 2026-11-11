"""
    HorseshoeFamily(dimension; scale=nothing, starts=nothing)

Fit a horseshoe prior with unit half-Cauchy local scales and a half-Cauchy global
scale. `scale` is the global scale's scale parameter. A numeric value fixes it;
`nothing` fits it through `log_scale`. Draws retain `coefficients`, `global_scale`,
and `local_scales` in one joint record.
Coefficient means and variances are undefined. Quantiles and event probabilities
provide finite elicitation targets.
"""
struct HorseshoeFamily{T,S} <: AbstractCompoundFamily
    dimension::Int
    scale::T
    starts::S
end
function HorseshoeFamily(n::Integer; scale=nothing, starts=nothing)
    n > 0 || throw(ArgumentError("horseshoe dimension must be positive"))
    _compound_positive(scale, :scale)
    HorseshoeFamily(n, scale, starts)
end

"""
    RegularizedHorseshoeFamily(dimension; scale=nothing, slab_scale=nothing,
                             slab_df=4.0, starts=nothing)

Fit a regularized horseshoe prior. The slab variance is `slab_scale^2 * slab_aux`,
where `slab_aux` has an inverse-Gamma law with shape and scale `slab_df / 2`.
Numeric `scale` and `slab_scale` values fix these parameters; `nothing` fits them
through `log_scale` and `log_slab_scale`. `slab_df` is fixed. Draws retain the
coefficients and their positive auxiliary scales in one joint record.
"""
struct RegularizedHorseshoeFamily{T,C,D,S} <: AbstractCompoundFamily
    dimension::Int
    scale::T
    slab_scale::C
    slab_df::D
    starts::S
end
function RegularizedHorseshoeFamily(
    n::Integer;
    scale=nothing,
    slab_scale=nothing,
    slab_df::Real=4.0,
    starts=nothing,
)
    n > 0 || throw(ArgumentError("horseshoe dimension must be positive"))
    _compound_positive(scale, :scale)
    _compound_positive(slab_scale, :slab_scale)
    _compound_positive(slab_df, :slab_df)
    RegularizedHorseshoeFamily(n, scale, slab_scale, slab_df, starts)
end

_compound_shape(f::Union{HorseshoeFamily,RegularizedHorseshoeFamily}) = (f.dimension,)
function _compound_starts(f::Union{HorseshoeFamily,RegularizedHorseshoeFamily})
    initial = (;)
    if f.scale === nothing
        initial = (; initial..., log_scale=log(0.1))
    end
    if f isa RegularizedHorseshoeFamily && f.slab_scale === nothing
        initial = (; initial..., log_slab_scale=0.0)
    end
    if f.starts === nothing
        return (initial,)
    elseif f.starts isa NamedTuple
        return (f.starts,)
    end
    f.starts
end
