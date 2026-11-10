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
const ArrayProduct = MeasureBase.ProductMeasure{<:AbstractArray}

function _product_transform(μ::NamedProduct, transform, lengthfn)
    marginals = MeasureBase.marginals(μ)
    lengths = map(lengthfn, marginals)
    ends = accumulate(+, values(lengths))
    ranges = NamedTuple{keys(marginals)}(
        map((stop, n) -> (stop-n+1):stop, ends, values(lengths)),
    )
    return Plaice.ProductVecTransform(map(transform, marginals), ranges, ())
end

function _product_transform(μ::ArrayProduct, transform, lengthfn)
    parts = MeasureBase.marginals(μ)
    isempty(parts) && throw(ArgumentError("an array prior needs at least one component"))
    base_size = size(MeasureBase.testvalue(first(parts)))
    all(p -> size(MeasureBase.testvalue(p)) == base_size, parts) ||
        throw(DimensionMismatch("array prior components need matching variate shapes"))
    lengths = map(lengthfn, parts)
    stops = accumulate(+, vec(lengths))
    ranges = reshape(map((stop, n) -> (stop-n+1):stop, stops, vec(lengths)), size(parts))
    Plaice.ProductVecTransform(map(transform, parts), ranges, base_size)
end
Plaice.to_vec(μ::ArrayProduct) = _product_transform(μ, Plaice.to_vec, Plaice.vec_length)
Plaice.to_unconstrained_vec(μ::ArrayProduct) =
    _product_transform(μ, Plaice.to_unconstrained_vec, Plaice.unconstrained_vec_length)
Plaice.from_vec(μ::ArrayProduct) = Plaice.inverse(Plaice.to_vec(μ))
Plaice.from_unconstrained_vec(μ::ArrayProduct) =
    Plaice.inverse(Plaice.to_unconstrained_vec(μ))
Plaice.vec_length(μ::ArrayProduct) = sum(Plaice.vec_length, MeasureBase.marginals(μ))
Plaice.unconstrained_vec_length(μ::ArrayProduct) =
    sum(Plaice.unconstrained_vec_length, MeasureBase.marginals(μ))

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

function _array_optics(μ, optics)
    parts = MeasureBase.marginals(μ)
    dimensions = length(size(MeasureBase.testvalue(first(parts))))
    result = Union{Nothing,VarNames.AbstractOptic}[]
    for index in CartesianIndices(parts), optic in optics(parts[index])
        if optic === nothing
            push!(result, nothing)
        else
            prefix = (ntuple(_ -> Colon(), dimensions)..., Tuple(index)...)
            push!(result, VarNames.Index(prefix, NamedTuple(), optic))
        end
    end
    result
end
Plaice.optic_vec(μ::ArrayProduct) = _array_optics(μ, Plaice.optic_vec)
Plaice.unconstrained_optic_vec(μ::ArrayProduct) =
    _array_optics(μ, Plaice.unconstrained_optic_vec)

const RecordMeasure = Credence.RecordMeasure
Plaice.to_vec(μ::RecordMeasure) =
    Plaice.to_vec(MeasureBase.parent(μ)) ∘ Credence.inverse(MeasureBase.gettransform(μ))
Plaice.from_vec(μ::RecordMeasure) =
    MeasureBase.gettransform(μ) ∘ Plaice.from_vec(MeasureBase.parent(μ))
Plaice.to_unconstrained_vec(μ::RecordMeasure) =
    Plaice.to_unconstrained_vec(MeasureBase.parent(μ)) ∘
    Credence.inverse(MeasureBase.gettransform(μ))
Plaice.from_unconstrained_vec(μ::RecordMeasure) =
    MeasureBase.gettransform(μ) ∘ Plaice.from_unconstrained_vec(MeasureBase.parent(μ))
Plaice.vec_length(μ::RecordMeasure) = Plaice.vec_length(MeasureBase.parent(μ))
Plaice.unconstrained_vec_length(μ::RecordMeasure) =
    Plaice.unconstrained_vec_length(MeasureBase.parent(μ))

function _record_optics(μ, optics)
    f = MeasureBase.gettransform(μ)
    labels = NamedTuple{Credence._factor_names(length(f.scopes))}(
        Credence._coordinate_labels(f.templates, Ref(0)),
    )
    leaves = VarNames.varname_and_value_leaves(f(labels))
    mapping =
        Dict(value => VarNames.varname_to_optic(reference) for (reference, value) in leaves)
    map(optics(MeasureBase.parent(μ))) do optic
        optic === nothing && return nothing
        mapping[optic(labels)]
    end
end
Plaice.optic_vec(μ::RecordMeasure) = _record_optics(μ, Plaice.optic_vec)
Plaice.unconstrained_optic_vec(μ::RecordMeasure) =
    _record_optics(μ, Plaice.unconstrained_optic_vec)

end
