"""
    DirichletLaplaceFamily(n; concentration=1/n, scale=nothing, starts=nothing)

Construct the Dirichlet–Laplace hierarchy with per-component concentration `a`,
allocation `ϕ ~ Dirichlet(fill(a, n))`, and `global_scale ~ Gamma(n*a, 2)` in
shape-scale form. Conditional coefficient scales are `scale * global_scale * ϕ`.
The output retains `coefficients`, `global_scale`, and `allocation`.

Set `concentration` or `scale` to `nothing` to fit it, using `log_concentration`
or `log_scale` in `starts`. Integrating both auxiliary variables gives independent
Gamma-mixture Laplace coefficients. The default concentration is `1/n`.
"""
struct DirichletLaplaceFamily{A,S,T} <: AbstractCompoundFamily
    dimension::Int
    concentration::A
    scale::S
    starts::T
end
function DirichletLaplaceFamily(
    n::Integer;
    concentration=inv(float(n)),
    scale=nothing,
    starts=nothing,
)
    n >= 2 || throw(ArgumentError("Dirichlet–Laplace dimension must be at least two"))
    _compound_positive(concentration, :concentration)
    _compound_positive(scale, :scale)
    DirichletLaplaceFamily(n, concentration, scale, starts)
end

_compound_shape(family::DirichletLaplaceFamily) = (family.dimension,)
function _compound_starts(family::DirichletLaplaceFamily)
    initial = NamedTuple()
    if family.concentration === nothing
        initial = (; initial..., log_concentration=(-log(float(family.dimension))))
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
