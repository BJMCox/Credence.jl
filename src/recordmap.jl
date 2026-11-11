# A coordinate regrouping keeps dependent factors intact inside a native pushforward.
struct FactorRecord{Names,S,T}
    scopes::S
    templates::T
end
struct FactorValues{F}
    record::F
end
inverse(f::FactorRecord) = FactorValues(f)
inverse(f::FactorValues) = f.record
with_logabsdet_jacobian(f::Union{FactorRecord,FactorValues}, x) = (f(x), 0)

_plain_record(x) = x
_plain_record(x::VarNames.VarNamedTuple) = _value_record(Tuple(pairs(x)))
_plain_record(x::AbstractArray) = map(_plain_record, x)
function _value_record(parts)
    refs = first.(parts)
    names = Tuple(unique(getsym.(refs)))
    if all(r -> getoptic(r) isa VarNames.Iden, refs)
        return NamedTuple{names}(last.(parts))
    end
    record = VarNames.VarNamedTuple(parts)
    NamedTuple{names}(map(name -> _plain_record(record[VarName{name}()]), names))
end
_factor_pairs(reference::VarName, value) = (reference=>value,)
function _factor_pairs(scope::Joint, value)
    refs = scope.references
    if value isa NamedTuple
        return map(ref -> ref=>getvalue(value, ref), refs)
    elseif length(refs) == 1
        return (only(refs)=>value,)
    end
    length(value) == length(refs) ||
        throw(DimensionMismatch("joint references must match factor coordinates"))
    map((ref, index) -> ref=>value[index], refs, Tuple(eachindex(value)))
end
function (f::FactorRecord{Names})(values) where {Names}
    parts =
        Tuple(Iterators.flatten(map(_factor_pairs, f.scopes, Tuple(Base.values(values)))))
    _value_record(parts)::NamedTuple{Names}
end
_factor_restore(template, value) = value
_factor_restore(template::AbstractArray, values) = reshape(collect(values), size(template))
_factor_restore(template::Tuple, values) = Tuple(values)
function _factor_restore(template::NamedTuple, values)
    NamedTuple{keys(template)}(
        map(
            k -> _factor_restore(getproperty(template, k), getproperty(values, k)),
            keys(template),
        ),
    )
end
_factor_extract(scope::VarName, template, record) = getvalue(record, scope)
function _factor_extract(scope::Joint, template, record)
    if template isa NamedTuple
        selected = _value_record(map(ref -> ref => getvalue(record, ref), scope.references))
        return _factor_restore(template, selected)
    end
    values = map(ref -> getvalue(record, ref), scope.references)
    length(scope.references) == 1 && return only(values)
    _factor_restore(template, values)
end
function (f::FactorValues)(record)
    values = map(
        (scope, template) -> _factor_extract(scope, template, record),
        f.record.scopes,
        f.record.templates,
    )
    NamedTuple{_factor_names(length(values))}(values)
end
_factor_names(n) = ntuple(i -> Symbol(:factor, i), n)

function _factor_template(law)
    try
        value = mean(law)
        all(isfinite, _flatten(value)) && return value
    catch error
        error isa Union{MethodError,ArgumentError,DomainError} || rethrow()
    end
    MeasureBase.testvalue(law)
end
_coordinate_labels(x, offset) = (offset[]+=1)
_coordinate_labels(x::Union{Tuple,NamedTuple,AbstractArray}, offset) =
    map(v -> _coordinate_labels(v, offset), x)

function _record_product(parts)
    scopes = first.(parts)
    laws = last.(parts)
    templates = map(_factor_template, laws)
    refs = Tuple(Iterators.flatten(map(_references, scopes)))
    names = Tuple(unique(getsym.(refs)))
    f = FactorRecord{names,typeof(scopes),typeof(templates)}(scopes, templates)
    labels = NamedTuple{_factor_names(length(laws))}(_coordinate_labels(templates, Ref(0)))
    isequal(inverse(f)(f(labels)), labels) ||
        throw(ArgumentError("factor bindings overlap or omit coordinates"))
    pushfwd(f, productmeasure(NamedTuple{keys(labels)}(laws)); varsize=())
end

const RecordMeasure = MeasureBase.PushforwardMeasure{<:FactorRecord}
function _law_reference(law::RecordMeasure, scope, reference)
    f = MeasureBase.gettransform(law)
    laws = Tuple(values(marginals(MeasureBase.parent(law))))
    for (factor, part, template) in zip(f.scopes, laws, f.templates)
        if factor isa VarName && subsumes(factor, reference)
            return _law_reference(part, factor, reference)
        elseif factor isa Joint
            index = findfirst(==(reference), factor.references)
            index === nothing && continue
            length(factor.references) == 1 && return part
            if template isa NamedTuple
                return _law_reference(part, nothing, reference)
            end
            return _law_optic(part, VarNames.Index((index,), NamedTuple()))
        end
    end
    throw(NoAnalytic(reference))
end

_vector(x::NamedTuple) = reduce(vcat, map(_vector, values(x)))
function _block_covariance(laws)
    blocks = map(laws) do law
        mean(law) isa Number ? reshape([var(law)], 1, 1) : _covariance(law)
    end
    lengths = map(b -> size(b, 1), blocks)
    stops = accumulate(+, lengths)
    T = promote_type(map(eltype, blocks)...)
    result = similar(first(blocks), T, last(stops), last(stops))
    fill!(result, zero(T))
    for (block, stop, n) in zip(blocks, stops, lengths)
        result[(stop-n+1):stop, (stop-n+1):stop] = block
    end
    result
end
_covariance(law::ProductMeasure{<:Union{Tuple,NamedTuple}}) =
    _block_covariance(Tuple(values(marginals(law))))
function _record_covariance(law::RecordMeasure, references)
    f = MeasureBase.gettransform(law)
    labels =
        NamedTuple{_factor_names(length(f.scopes))}(_coordinate_labels(f.templates, Ref(0)))
    record = f(labels)
    indices = reduce(vcat, map(ref -> _vector(getvalue(record, ref)), references))
    _covariance(MeasureBase.parent(law))[indices, indices]
end
_joint_covariance(law::RecordMeasure, scope, target, parts) =
    _record_covariance(law, target.references)
function _analytic_on(
    s::Union{Mean,Std,CovarianceMatrix,CorrelationMatrix},
    law::RecordMeasure,
    scope,
    target::VarName,
)
    value = getvalue(MeasureBase.gettransform(law)(mean(MeasureBase.parent(law))), target)
    s isa Mean && return _estimate(value)
    covariance = _record_covariance(law, (target,))
    if s isa CovarianceMatrix
        return _estimate(covariance)
    elseif s isa CorrelationMatrix
        scales = sqrt.(diag(covariance))
        return _estimate(covariance ./ (scales*scales'))
    end
    _estimate(
        value isa Number ? sqrt(only(covariance)) :
        reshape(sqrt.(diag(covariance)), size(value)),
    )
end

function MeasureBase.rand_impl(
    ctx::GenContext,
    power::MeasureBase.PowerMeasure{<:RecordMeasure},
)
    law = MeasureBase.pwr_base(power)
    map(
        MeasureBase.gettransform(law),
        MeasureBase.rand_impl(ctx, MeasureBase.parent(law)^MeasureBase.pwr_axes(power)),
    )
end
MeasureBase.logdensities(law::RecordMeasure, draws::AbstractArray{<:NamedTuple}) =
    map(x -> MeasureBase.logdensityof(law, x), draws)
function MeasureBase.logdensityof_impl(
    power::MeasureBase.PowerMeasure{<:RecordMeasure},
    draws::AbstractArray{<:NamedTuple},
)
    size(draws) == map(length, MeasureBase.pwr_axes(power)) ||
        throw(DimensionMismatch("draws must match the power shape"))
    sum(x -> MeasureBase.logdensityof(MeasureBase.pwr_base(power), x), draws; init=0.0)
end
function MeasureBase.logdensity_def(
    power::MeasureBase.PowerMeasure{<:RecordMeasure},
    draws::AbstractArray{<:NamedTuple},
)
    size(draws) == map(length, MeasureBase.pwr_axes(power)) ||
        throw(DimensionMismatch("draws must match the power shape"))
    sum(x -> MeasureBase.logdensity_def(MeasureBase.pwr_base(power), x), draws; init=0.0)
end
