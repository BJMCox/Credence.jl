"""
    FunctionalHorseshoeFamily(design, baseline; baseline_scale, scale=nothing,
                             a=0.5, b=0.5, starts=nothing)

Shrink a function `design * coefficients` towards the column space of `baseline`.
The design must have full column rank and contain the baseline subspace.
Redundant baseline columns are allowed. At least one direction must lie outside
the baseline. The output retains `coefficients` and `shrinkage`.

The shrinkage weight follows `Beta(a, b)` and gives
`tau = sqrt((1 - shrinkage) / shrinkage)`. Conditional deviations have scale
`scale * tau` in an orthonormal observation-space basis. Set `scale=nothing`
to fit it, using `log_scale` in `starts`.

`baseline_scale` sets a proper zero-mean Gaussian law on the baseline projection,
in observation-space units. This completes the improper baseline directions in
Shin, Bhattacharya, and Johnson's functional horseshoe prior (equations 4–5).
Treat the stored design, baseline, and cached geometry arrays as read-only.
Construct a new family when changing its geometry.
"""
struct FunctionalHorseshoeFamily{X,B,C,S,A,D,T,L,P,V} <: AbstractCompoundFamily
    design::X
    baseline::B
    baseline_scale::C
    scale::S
    a::A
    b::D
    starts::T
    loading::L
    precision::P
    determinant::V
    baseline_dimension::Int
end

function FunctionalHorseshoeFamily(
    design::AbstractMatrix{<:Real},
    baseline::AbstractVecOrMat{<:Real};
    baseline_scale,
    scale=nothing,
    a=0.5,
    b=0.5,
    starts=nothing,
)
    _compound_positive(baseline_scale, :baseline_scale)
    baseline_scale === nothing && throw(ArgumentError("baseline_scale must be fixed"))
    _compound_positive(scale, :scale)
    _compound_positive(a, :a)
    _compound_positive(b, :b)
    (a === nothing || b === nothing) && throw(ArgumentError("a and b must be fixed"))
    n, k = size(design)
    n >= k > 0 || throw(ArgumentError("design needs at least as many rows as columns"))
    size(baseline, 1) == n ||
        throw(DimensionMismatch("baseline and design rows must match"))
    X = Matrix(float.(design))
    B = reshape(collect(float.(baseline)), n, :)
    all(isfinite, X) && all(isfinite, B) ||
        throw(ArgumentError("design and baseline must be finite"))
    tolerance = max(n, k) * eps(real(float(one(eltype(X)))))
    singular = svdvals(X)
    minimum(singular) > tolerance * maximum(singular) ||
        throw(ArgumentError("design must have full column rank"))
    decomposition = qr(X)
    Q = decomposition.Q * Matrix{eltype(X)}(I, n, k)
    R = Matrix(decomposition.R)
    coordinates = Q' * B
    norm(B - Q*coordinates) <= 8tolerance * norm(B) ||
        throw(ArgumentError("baseline must lie within the design column space"))
    U, dimension = if isempty(B)
        (Matrix{eltype(X)}(I, k, k), 0)
    else
        decomposition = svd(coordinates; full=true)
        threshold =
            max(size(coordinates)...) *
            eps(eltype(decomposition.S)) *
            maximum(decomposition.S)
        (decomposition.U, count(>(threshold), decomposition.S))
    end
    dimension < k || throw(ArgumentError("baseline leaves no direction to shrink"))
    loading = R \ U
    precision = U' * R
    determinant = logabsdet(loading)
    FunctionalHorseshoeFamily(
        X,
        B,
        baseline_scale,
        scale,
        a,
        b,
        starts,
        loading,
        precision,
        determinant,
        dimension,
    )
end

function _compound_validate(family::FunctionalHorseshoeFamily)
    rebuilt = FunctionalHorseshoeFamily(
        family.design,
        family.baseline;
        baseline_scale=family.baseline_scale,
        scale=family.scale,
        a=family.a,
        b=family.b,
        starts=family.starts,
    )
    isequal(family.loading, rebuilt.loading) &&
    isequal(family.precision, rebuilt.precision) &&
    isequal(family.determinant, rebuilt.determinant) &&
    family.baseline_dimension == rebuilt.baseline_dimension || throw(
        ArgumentError("functional horseshoe geometry changed; construct a new family"),
    )
    nothing
end

_compound_shape(family::FunctionalHorseshoeFamily) = (size(family.design, 2),)
function _compound_starts(family::FunctionalHorseshoeFamily)
    initial = NamedTuple()
    if family.scale === nothing
        initial = (log_scale=0.0,)
    end
    if family.starts === nothing
        return (initial,)
    elseif family.starts isa NamedTuple
        return (family.starts,)
    end
    family.starts
end

# Fixed geometry is factored once. Each draw changes only the column scales.
struct FunctionalFactor{F,S}
    family::F
    scales::S
end
Base.:*(factor::FunctionalFactor, x::AbstractVector) =
    factor.family.loading * (factor.scales .* x)
Base.:\(factor::FunctionalFactor, x::AbstractVector) =
    (factor.family.precision * x) ./ factor.scales
LinearAlgebra.logabsdet(factor::FunctionalFactor) =
    (factor.family.determinant[1] + sum(log, factor.scales), factor.family.determinant[2])
