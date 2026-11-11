"""
    MultiplicativeGammaFamily(rows, columns; first_shape=2.0, later_shape=3.0,
                              local_df=3.0, scale=nothing, starts=nothing)

Construct a finite multiplicative Gamma hierarchy for a loading matrix. Local
precisions have shape-rate parameters `local_df/2, local_df/2`. Independent Gamma
increments have unit rate, with shape `first_shape` for the first column and
`later_shape` for later columns. Their cumulative products form column precisions.
Conditional loading variances are `scale^2 / (local_precision * column_precision)`.
The output retains `coefficients`, `local_precisions`, and `increments`.

Set a parameter to `nothing` to fit it through its `log_` starting coordinate.
All finite positive shapes define a proper finite prior. Expected loading variances
decrease across columns when `local_df > 2`, `first_shape > 1`, and `later_shape > 2`.
Individual increments can be below one, so shrinkage varies between draws.
"""
struct MultiplicativeGammaFamily{A,B,N,S,T} <: AbstractCompoundFamily
    rows::Int
    columns::Int
    first_shape::A
    later_shape::B
    local_df::N
    scale::S
    starts::T
end
function MultiplicativeGammaFamily(
    rows::Integer,
    columns::Integer;
    first_shape=2.0,
    later_shape=3.0,
    local_df=3.0,
    scale=nothing,
    starts=nothing,
)
    rows > 0 && columns > 0 ||
        throw(ArgumentError("loading matrix dimensions must be positive"))
    for (name, value) in pairs((; first_shape, later_shape, local_df, scale))
        _compound_positive(value, name)
    end
    MultiplicativeGammaFamily(
        rows,
        columns,
        first_shape,
        later_shape,
        local_df,
        scale,
        starts,
    )
end

_compound_shape(family::MultiplicativeGammaFamily) = (family.rows, family.columns)
function _compound_starts(family::MultiplicativeGammaFamily)
    initial = NamedTuple()
    for (name, value) in pairs((first_shape=2.0, later_shape=3.0, local_df=3.0, scale=1.0))
        if getproperty(family, name) === nothing
            initial = merge(initial, NamedTuple{(Symbol(:log_, name),)}((log(value),)))
        end
    end
    if family.starts === nothing
        return (initial,)
    elseif family.starts isa NamedTuple
        return (family.starts,)
    end
    family.starts
end
