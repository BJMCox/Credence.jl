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

const TruncatedPrior = Union{
    MeasureBase.TruncatedMeasure,
    MeasureBase.NormalizedMeasure{<:MeasureBase.TruncatedMeasure},
}
function Plaice.to_unconstrained_vec(μ::TruncatedPrior)
    mass = MeasureBase.massof(μ)
    lower = MeasureBase.invsmf(μ, 0)
    upper = MeasureBase.invsmf(μ, mass)
    Plaice.VectWrap(Plaice.Untruncate(lower, upper))
end
Plaice.from_unconstrained_vec(μ::TruncatedPrior) =
    Plaice.inverse(Plaice.to_unconstrained_vec(μ))
Plaice.to_vec(::TruncatedPrior) = Plaice.VectWrap(Plaice.TypedIdentity())
Plaice.from_vec(μ::TruncatedPrior) = Plaice.inverse(Plaice.to_vec(μ))
Plaice.vec_length(::TruncatedPrior) = 1
Plaice.unconstrained_vec_length(::TruncatedPrior) = 1
Plaice.optic_vec(::TruncatedPrior) = [VarNames.Iden()]
Plaice.unconstrained_optic_vec(::TruncatedPrior) = [VarNames.Iden()]

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

const CompoundMeasure = Credence.CompoundMeasure
include("compound_simplex.jl")

struct CompoundFromVec{M,F}
    law::M
    parent_transform::F
end
struct CompoundToVec{M,F}
    law::M
    parent_transform::F
end
_inverse_parent_transform(::typeof(Plaice.from_vec)) = Plaice.to_vec
_inverse_parent_transform(::typeof(Plaice.to_vec)) = Plaice.from_vec
_inverse_parent_transform(::typeof(Plaice.from_unconstrained_vec)) =
    Plaice.to_unconstrained_vec
_inverse_parent_transform(::typeof(Plaice.to_unconstrained_vec)) =
    Plaice.from_unconstrained_vec
function (chart::CompoundFromVec)(coordinates)
    base = _compound_parent_chart(chart.law, chart.parent_transform)(coordinates)
    MeasureBase.gettransform(chart.law)(base)
end
function (chart::CompoundToVec)(point)
    haskey(point, :allocation) && _check_compound_allocation(point.allocation)
    base = Credence.inverse(MeasureBase.gettransform(chart.law))(point)
    _compound_parent_chart(chart.law, chart.parent_transform)(base)
end
Plaice.inverse(chart::CompoundFromVec) =
    CompoundToVec(chart.law, _inverse_parent_transform(chart.parent_transform))
Plaice.inverse(chart::CompoundToVec) =
    CompoundFromVec(chart.law, _inverse_parent_transform(chart.parent_transform))
function Plaice.with_logabsdet_jacobian(chart::CompoundFromVec, coordinates)
    base, parent_jacobian = Plaice.with_logabsdet_jacobian(
        _compound_parent_chart(chart.law, chart.parent_transform),
        coordinates,
    )
    point, coefficient_jacobian =
        Plaice.with_logabsdet_jacobian(MeasureBase.gettransform(chart.law), base)
    point, parent_jacobian + coefficient_jacobian
end
function Plaice.with_logabsdet_jacobian(chart::CompoundToVec, point)
    haskey(point, :allocation) && _check_compound_allocation(point.allocation)
    base, coefficient_jacobian = Plaice.with_logabsdet_jacobian(
        Credence.inverse(MeasureBase.gettransform(chart.law)),
        point,
    )
    coordinates, parent_jacobian = Plaice.with_logabsdet_jacobian(
        _compound_parent_chart(chart.law, chart.parent_transform),
        base,
    )
    coordinates, coefficient_jacobian + parent_jacobian
end

for (to, from) in ((:to_vec, :from_vec), (:to_unconstrained_vec, :from_unconstrained_vec))
    @eval Plaice.$to(μ::CompoundMeasure) = CompoundToVec(μ, Plaice.$to)
    # Applying the maps explicitly avoids a temporary composed closure whose
    # captured arrays cannot always acquire an Enzyme shadow.
    @eval Plaice.$from(μ::CompoundMeasure) = CompoundFromVec(μ, Plaice.$from)
end
Plaice.vec_length(μ::CompoundMeasure) = Plaice.vec_length(MeasureBase.parent(μ))
Plaice.unconstrained_vec_length(μ::CompoundMeasure) =
    Plaice.unconstrained_vec_length(MeasureBase.parent(μ))
# The first coordinates map jointly to the coefficient field. Their auxiliary
# coordinates retain their names through the transformation.
function _compound_optics(μ, optics, lengthfn)
    parent = MeasureBase.parent(μ)
    count = lengthfn(first(values(MeasureBase.marginals(parent))))
    result = copy(optics(parent))
    result[1:count] .= nothing
    result
end
Plaice.optic_vec(μ::CompoundMeasure) =
    _compound_optics(μ, Plaice.optic_vec, Plaice.vec_length)
Plaice.unconstrained_optic_vec(μ::CompoundMeasure) =
    _compound_optics(μ, Plaice.unconstrained_optic_vec, Plaice.unconstrained_vec_length)

end
