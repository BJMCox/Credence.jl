struct AllocationToVec end
struct AllocationFromVec end
Plaice.inverse(::AllocationToVec) = AllocationFromVec()
Plaice.inverse(::AllocationFromVec) = AllocationToVec()

function _check_compound_allocation(weights)
    all(x -> isfinite(x) && x > 0, weights) || throw(
        ArgumentError(
            "allocation weights must be positive and finite; use higher-precision family parameters and a matching GenContext for sparse allocations",
        ),
    )
    total = sum(weights)
    isapprox(total, one(total)) ||
        throw(ArgumentError("allocation weights must sum to one"))
    nothing
end
function (::AllocationToVec)(weights::AbstractVector)
    _check_compound_allocation(weights)
    log.(view(weights, 1:(length(weights)-1))) .- log(last(weights))
end

function (::AllocationFromVec)(coordinates::AbstractVector)
    isempty(coordinates) && return fill(one(eltype(coordinates)), 1)
    offset = max(maximum(coordinates), zero(eltype(coordinates)))
    weights = exp.(vcat(coordinates, zero(eltype(coordinates))) .- offset)
    weights ./ sum(weights)
end

function Plaice.with_logabsdet_jacobian(chart::AllocationToVec, weights)
    chart(weights), -sum(log, weights)
end
function Plaice.with_logabsdet_jacobian(chart::AllocationFromVec, coordinates)
    weights = chart(coordinates)
    weights, sum(log, weights)
end

_compound_parent_chart(law, constructor) = constructor(MeasureBase.parent(law))
function _compound_parent_chart(law, ::typeof(Plaice.to_unconstrained_vec))
    chart = Plaice.to_unconstrained_vec(MeasureBase.parent(law))
    if !haskey(chart.transforms, :allocation)
        return chart
    end
    transforms = (; chart.transforms..., allocation=AllocationToVec())
    Plaice.ProductVecTransform(transforms, chart.ranges, chart.base_size)
end
_compound_parent_chart(law, ::typeof(Plaice.from_unconstrained_vec)) =
    Plaice.inverse(_compound_parent_chart(law, Plaice.to_unconstrained_vec))
