_moment_order(statistic) = 0
_moment_order(::Mean) = 1
_moment_order(::Union{Std,Covariance,Correlation,CovarianceMatrix,CorrelationMatrix}) = 2

function _coefficient_indices(family, scope, reference)
    optic = getoptic(reference)
    if scope isa VarName
        subsumes(scope, reference) || return nothing
        prefix = getoptic(scope)
        while !(prefix isa VarNames.Iden)
            optic = VarNames.otail(optic)
            prefix = VarNames.otail(prefix)
        end
        optic isa VarNames.Property{:coefficients} || return nothing
        optic = VarNames.otail(optic)
    else
        getsym(reference) == :coefficients || return nothing
    end
    shape = _compound_shape(family)
    selected = optic(reshape(1:prod(shape), shape))
    selected isa Integer && return [selected]
    vec(collect(selected))
end

function _check_moment(statistic::On{<:Union{VarName,Joint}}, law::CompoundMeasure, scope)
    order = _moment_order(statistic.statistic)
    order == 0 && return nothing
    transform = MeasureBase.gettransform(law)
    for reference in _references(statistic.target)
        _covers(law, scope, reference) || continue
        indices = _coefficient_indices(transform.family, scope, reference)
        if indices === nothing
            marginal = _law_reference(law, scope, reference)
            required = if order == 1
                Mean()
            else
                Std()
            end
            try
                evaluate(required, marginal)
            catch error
                error isa InterruptException && rethrow()
                throw(
                    ArgumentError(
                        "the auxiliary moment cannot be established analytically: $(sprint(showerror, error))",
                    ),
                )
            end
            continue
        end
        _compound_moment_exists(transform.family, transform.parameters, order, indices) ||
            throw(
                ArgumentError(
                    "this coefficient statistic requires finite absolute moments of order $order; use quantiles or event probabilities",
                ),
            )
    end
    nothing
end
function _check_moment(statistic::On, law::ProductMeasure{<:AbstractArray}, scope::VarName)
    parts = MeasureBase.marginals(law)
    for index in CartesianIndices(parts)
        child = VarNames.Index(Tuple(index), NamedTuple(), VarNames.Iden())
        _check_moment(statistic, parts[index], VarNames.append_optic(scope, child))
    end
    nothing
end
function _check_moment(statistic::On, law::ProductMeasure{<:NamedTuple}, scope)
    for (name, part) in pairs(MeasureBase.marginals(law))
        child_scope = if scope isa VarName
            VarNames.append_optic(scope, VarNames.Property{name}(VarNames.Iden()))
        else
            VarName{name}()
        end
        _check_moment(statistic, part, child_scope)
    end
    nothing
end
function _check_moment(statistic::On, law::RecordMeasure, scope)
    transform = MeasureBase.gettransform(law)
    for (part_scope, part) in
        zip(transform.scopes, values(MeasureBase.marginals(MeasureBase.parent(law))))
        _check_moment(statistic, part, part_scope)
    end
    nothing
end

_compound_moment_exists(::Union{HorseshoeFamily,GroupHorseshoeFamily}, p, order, indices) =
    false
_compound_moment_exists(f::RegularizedHorseshoeFamily, p, order, indices) =
    f.slab_df + 1 > order
_compound_moment_exists(
    ::Union{DirichletLaplaceFamily,NormalGammaFamily},
    p,
    order,
    indices,
) = true
_compound_moment_exists(f::GeneralizedDoubleParetoFamily, p, order, indices) =
    _compound_parameter(f, p, Val(:shape)) > order
_compound_moment_exists(f::R2D2Family, p, order, indices) =
    (1 - _r2d2_mean(f, p)) * _compound_parameter(f, p, Val(:precision_r2)) > order / 2
function _compound_moment_exists(f::FunctionalHorseshoeFamily, p, order, indices)
    contrast = view(f.loading, indices, (f.baseline_dimension+1):size(f.loading, 2))
    !any(!iszero, contrast) || 2f.a > order
end
function _compound_moment_exists(f::MultiplicativeGammaFamily, p, order, indices)
    _compound_parameter(f, p, Val(:local_df)) > order &&
    _compound_parameter(f, p, Val(:first_shape)) > order / 2 || return false
    all(index -> index <= f.rows, indices) ||
        _compound_parameter(f, p, Val(:later_shape)) > order / 2
end
