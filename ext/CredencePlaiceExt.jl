module CredencePlaiceExt

using Credence
using MeasureBase: MeasureBase
using Plaice: Plaice
using VarNames: VarNames

# Wrapped laws retain their support charts rather than using a transport that
# whitens the prior or changes the original-space coordinates.
for f in (
    :to_vec,
    :from_vec,
    :to_unconstrained_vec,
    :from_unconstrained_vec,
    :vec_length,
    :unconstrained_vec_length,
    :optic_vec,
    :unconstrained_optic_vec,
)
    @eval Plaice.$f(μ::MeasureBase.AsMeasure) = Plaice.$f(μ.obj)
end

const NamedProduct = MeasureBase.ProductMeasure{<:NamedTuple}

function _product_transform(μ::NamedProduct, transform, lengthfn)
    marginals = MeasureBase.marginals(μ)
    lengths = map(lengthfn, marginals)
    ends = accumulate(+, values(lengths))
    ranges = NamedTuple{keys(marginals)}(
        map((stop, n) -> (stop-n+1):stop, ends, values(lengths)),
    )
    return Plaice.ProductVecTransform(map(transform, marginals), ranges, ())
end

Plaice.to_vec(μ::NamedProduct) = _product_transform(μ, Plaice.to_vec, Plaice.vec_length)
Plaice.to_unconstrained_vec(μ::NamedProduct) =
    _product_transform(μ, Plaice.to_unconstrained_vec, Plaice.unconstrained_vec_length)
Plaice.from_vec(μ::NamedProduct) = Plaice.inverse(Plaice.to_vec(μ))
Plaice.from_unconstrained_vec(μ::NamedProduct) =
    Plaice.inverse(Plaice.to_unconstrained_vec(μ))
Plaice.vec_length(μ::NamedProduct) = sum(Plaice.vec_length, MeasureBase.marginals(μ))
Plaice.unconstrained_vec_length(μ::NamedProduct) =
    sum(Plaice.unconstrained_vec_length, MeasureBase.marginals(μ))

_prefix(::Val{name}, optic::VarNames.AbstractOptic) where {name} =
    VarNames.Property{name}(optic)
_prefix(::Val, ::Nothing) = nothing

function _product_optics(μ::NamedProduct, optics)
    marginals = MeasureBase.marginals(μ)
    parts = map(keys(marginals), values(marginals)) do name, marginal
        map(optic -> _prefix(Val(name), optic), optics(marginal))
    end
    return reduce(vcat, parts; init=Union{Nothing,VarNames.AbstractOptic}[])
end

Plaice.optic_vec(μ::NamedProduct) = _product_optics(μ, Plaice.optic_vec)
Plaice.unconstrained_optic_vec(μ::NamedProduct) =
    _product_optics(μ, Plaice.unconstrained_optic_vec)

end
