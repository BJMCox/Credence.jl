abstract type AbstractCompoundFamily <: AbstractFamily end

function _compound_positive(value, name)
    value === nothing ||
        (isfinite(value) && value > 0) ||
        throw(ArgumentError("$name must be positive and finite"))
    nothing
end
function _compound_parameter(family, parameters, ::Val{name}) where {name}
    fixed = getproperty(family, name)
    fixed === nothing || return fixed
    exp(getproperty(parameters, Symbol(:log_, name)))
end
function _compound_base end
function _compound_linear end
function _compound_shape end
function _compound_starts end
_compound_validate(family) = nothing
_compound_validate(family::IIDFamily) = _compound_validate(family.family)

# Retaining the auxiliary scales makes the hierarchy invertible and its joint
# density available. Dropping them would require a separate marginal density.
struct CompoundTransform{F,P}
    family::F
    parameters::P
end
struct CompoundInverse{T}
    transform::T
end
inverse(transform::CompoundTransform) = CompoundInverse(transform)
inverse(transform::CompoundInverse) = transform.transform
function (transform::CompoundTransform)(point)
    factor = _compound_linear(transform.family, transform.parameters, point)
    coefficients = reshape(factor * point.z, _compound_shape(transform.family))
    (; coefficients, Base.structdiff(point, (z=nothing,))...)
end
function (transform::CompoundInverse)(point)
    forward = transform.transform
    auxiliary = Base.structdiff(point, (coefficients=nothing,))
    factor =
        _compound_linear(forward.family, forward.parameters, (; z=nothing, auxiliary...))
    (; z=factor \ vec(point.coefficients), auxiliary...)
end
function with_logabsdet_jacobian(transform::CompoundTransform, point)
    factor = _compound_linear(transform.family, transform.parameters, point)
    (transform(point), first(logabsdet(factor)))
end
function with_logabsdet_jacobian(transform::CompoundInverse, point)
    base = transform(point)
    forward = transform.transform
    factor = _compound_linear(forward.family, forward.parameters, base)
    (base, -first(logabsdet(factor)))
end

struct CoefficientTransform{L,S}
    factor::L
    shape::S
end
struct CoefficientInverse{T}
    transform::T
end
inverse(transform::CoefficientTransform) = CoefficientInverse(transform)
inverse(transform::CoefficientInverse) = transform.transform
(transform::CoefficientTransform)(z) = reshape(transform.factor * z, transform.shape)
(transform::CoefficientInverse)(x) = transform.transform.factor \ vec(x)
with_logabsdet_jacobian(transform::CoefficientTransform, z) =
    (transform(z), first(logabsdet(transform.factor)))
with_logabsdet_jacobian(transform::CoefficientInverse, x) =
    (transform(x), -first(logabsdet(transform.transform.factor)))

struct CompoundKernel{T,M}
    transform::T
    origin::M
end
function (kernel::CompoundKernel)(auxiliary)
    transform = kernel.transform
    factor = _compound_linear(
        transform.family,
        transform.parameters,
        (; z=nothing, auxiliary...),
    )
    coefficient_map = CoefficientTransform(factor, _compound_shape(transform.family))
    coefficient_law = pushfwd(
        coefficient_map,
        MeasureBase.marginals(kernel.origin).z;
        varsize=_compound_shape(transform.family),
    )
    productmeasure((coefficients=coefficient_law,))
end

function _family_candidate(id, family::AbstractCompoundFamily)
    function build(parameters)
        base = _compound_base(family, parameters)
        auxiliary =
            productmeasure(Base.structdiff(MeasureBase.marginals(base), (z=nothing,)))
        kernel = CompoundKernel(CompoundTransform(family, parameters), base)
        # Only the coefficient block changes coordinates. In particular, the
        # auxiliary simplex keeps its reduced-coordinate reference measure.
        MeasureBase.mbind(kernel, auxiliary, merge)
    end
    Candidate(id, build, _compound_starts(family); family)
end

const CompoundMeasure = MeasureBase.Bind{<:CompoundKernel}
MeasureBase.parent(law::CompoundMeasure) = law.f_β.origin
MeasureBase.gettransform(law::CompoundMeasure) = law.f_β.transform
# Every conditional coefficient law and every auxiliary factor is normalized.
MeasureBase.massof(law::CompoundMeasure) =
    one(MeasureBase.massof(MeasureBase.boundmeasure(law)))
_positive_auxiliary(value::Number) = isfinite(value) && value > 0
_positive_auxiliary(value) = all(_positive_auxiliary, value)
_positive_auxiliary(value::NamedTuple) = all(map(_positive_auxiliary, values(value)))

# Evaluate scalar factors eagerly. Lazy broadcast reductions through a reconstructed
# product chart lose their array rank during Enzyme inference.
_compound_point_logdensity(law, point) = logdensityof(law, point)
function _compound_point_logdensity(
    law::ProductMeasure{<:AbstractArray},
    point::AbstractArray{<:Number},
)
    parts = MeasureBase.marginals(law)
    size(parts) == size(point) || throw(DimensionMismatch("product point shape mismatch"))
    isempty(parts) && return zero(eltype(point))
    total = _compound_point_logdensity(first(parts), first(point))
    for index in Iterators.drop(eachindex(parts, point), 1)
        total += _compound_point_logdensity(parts[index], point[index])
    end
    total
end
function _compound_point_logdensity(law::ProductMeasure{<:NamedTuple}, point::NamedTuple)
    parts = MeasureBase.marginals(law)
    ordered = NamedTuple{keys(parts)}(point)
    sum(map(_compound_point_logdensity, values(parts), values(ordered)))
end

function MeasureBase.logdensityof_impl(law::CompoundMeasure, point::NamedTuple)
    names = keys(MeasureBase.marginals(MeasureBase.boundmeasure(law)))
    auxiliary = NamedTuple{names}(point)
    valid = _positive_auxiliary(auxiliary)
    for name in (:r2, :shrinkage)
        if haskey(auxiliary, name)
            valid &= getproperty(auxiliary, name) < 1
        end
    end
    valid || return oftype(float(first(point.coefficients)), -Inf)
    transform = MeasureBase.gettransform(law)
    size(point.coefficients) == _compound_shape(transform.family) ||
        throw(DimensionMismatch("coefficient point shape mismatch"))
    factor = _compound_linear(
        transform.family,
        transform.parameters,
        (; z=nothing, auxiliary...),
    )
    basepoint = (; z=factor \ vec(point.coefficients), auxiliary...)
    _compound_point_logdensity(MeasureBase.parent(law), basepoint) -
    first(logabsdet(factor))
end
_law_optic(law::CompoundMeasure, ::VarNames.Iden) = law
function _law_optic(law::CompoundMeasure, optic)
    optic isa VarNames.Property{:coefficients} && throw(NoAnalytic(optic))
    _law_optic(MeasureBase.parent(law), optic)
end
