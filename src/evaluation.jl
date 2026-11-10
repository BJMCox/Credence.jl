_statistic(s::On) = s.statistic
_statistic(s) = s
_analytic(s, law, scope) = evaluate(s, law)
_analytic(s::On, law, scope) = _analytic_on(s.statistic, law, scope, s.target)
_sample_target(draw, scope, s) = draw
_sample_target(draw, scope, s::On) = _draw_target(draw, scope, s.target)

_sampling_seed(rng::AbstractRNG) = rand(rng, UInt64)
_sampling_stream(rng::AbstractRNG, seed) = Xoshiro(seed)
function _sampling_seed(ctx::GenContext)
    words = allocate_array(get_compute_unit(ctx), UInt32, (2,))
    Random.rand!(ctx, words)
    values = Array(words)
    UInt64(values[1]) | UInt64(values[2]) << 32
end
function _sampling_stream(ctx::GenContext, seed)
    rng = deepcopy(get_rng(ctx))
    Random.seed!(rng, seed)
    GenContext{get_precision(ctx)}(get_compute_unit(ctx), rng)
end

struct EvaluationFailure{E}
    error::E
end

# This batch boundary allows device-specific drawing without changing belief binding.
_draw_batch(rng, law, n) = [rand(rng, law) for _ in 1:n]

function _estimates(
    beliefs,
    law,
    scope;
    sampling=nothing,
    rng=nothing,
    force_sampling=false,
)
    initial = map(beliefs) do belief
        s = belief.statistic
        _applicable(s, law, scope) || return nothing
        force_sampling && return NoAnalytic(s)
        try
            _analytic(s, law, scope)
        catch error
            error isa InterruptException && rethrow()
            if error isa NoAnalytic ||
               (error isa MethodError && !(_statistic(s) isa Statistic))
                NoAnalytic(s)
            else
                EvaluationFailure(error)
            end
        end
    end
    any(x -> x isa NoAnalytic, initial) || return initial
    if !(sampling isa MonteCarlo)
        return map(x -> x isa NoAnalytic ? EvaluationFailure(x) : x, initial)
    end
    rng === nothing && throw(ArgumentError("sampled evaluation requires an explicit RNG"))
    draws = try
        _draw_batch(rng, law, sampling.n)
    catch error
        error isa InterruptException && rethrow()
        return map(x -> x isa NoAnalytic ? EvaluationFailure(error) : x, initial)
    end
    projected = Dict{Any,Any}()
    map(beliefs, initial) do belief, estimate
        estimate isa NoAnalytic || return estimate
        s = belief.statistic
        try
            key = s isa On ? _targetid(s.target) : nothing
            values = get!(projected, key) do
                map(draw -> _sample_target(draw, scope, s), draws)
            end
            joint = s isa On && s.target isa Joint
            _sample_estimate(_statistic(s), values, sampling; joint)
        catch error
            error isa InterruptException && rethrow()
            EvaluationFailure(error)
        end
    end
end

_vector(x::Number) = [x]
_vector(x::AbstractArray) = vec(x)
_vector(x::Tuple) = reduce(vcat, map(_vector, x))
_sample_matrix(values) = reduce(hcat, map(_vector, values))
_shape_value(value, template::Number) = only(value)
_shape_value(value, template::AbstractArray) = reshape(value, size(template))
_shape_value(value, template::Tuple) = value
_zscore(alpha) = MeasureBase.invsmf(StdNormal(), 1-alpha/2)

function _sample_estimate(s::Probability, values, plan; joint=false)
    hits = count(values) do value
        joint ? s.event(value...) : s.event(value)
    end
    value = hits / length(values)
    radius = sqrt(log(2/plan.alpha)/(2length(values)))
    Estimate(
        value;
        lower=max(0, value-radius),
        upper=min(1, value+radius),
        evidence=:statistical,
        n=length(values),
    )
end

function _sample_estimate(::Mean, values, plan; joint=false)
    matrix = _sample_matrix(values)
    point = vec(mean(matrix; dims=2))
    errors = if length(values) > 1
        vec(std(matrix; dims=2)) / sqrt(length(values))
    else
        fill(Inf, length(point))
    end
    radius = _zscore(plan.alpha/length(point)) .* errors
    template = first(values)
    Estimate(
        _shape_value(point, template);
        lower=_shape_value(point-radius, template),
        upper=_shape_value(point+radius, template),
        evidence=:estimated,
        n=length(values),
    )
end

function _sample_estimate(s::Quantile, values, plan; joint=false)
    matrix = sort(_sample_matrix(values); dims=2)
    n = length(values)
    radius = sqrt(log(2size(matrix, 1)/plan.alpha)/(2n))
    point = matrix[:, clamp(ceil(Int, n*s.p), 1, n)]
    lower =
        s.p > radius ? matrix[:, max(1, ceil(Int, n*(s.p-radius)))] :
        fill(-Inf, length(point))
    upper = s.p+radius < 1 ? matrix[:, ceil(Int, n*(s.p+radius))] : fill(Inf, length(point))
    template = first(values)
    Estimate(
        _shape_value(point, template);
        lower=_shape_value(lower, template),
        upper=_shape_value(upper, template),
        evidence=:statistical,
        n,
    )
end

_sample_point(::Std, matrix) = vec(std(matrix; dims=2))
_sample_point(::CovarianceMatrix, matrix) = cov(matrix; dims=2)
_sample_point(::CorrelationMatrix, matrix) = cor(matrix; dims=2)
function _sample_point(::Covariance, matrix)
    size(matrix, 1) == 2 || throw(DimensionMismatch("Covariance requires two scalars"))
    cov(matrix; dims=2)[1, 2]
end
function _sample_point(::Correlation, matrix)
    size(matrix, 1) == 2 || throw(DimensionMismatch("Correlation requires two scalars"))
    cor(matrix; dims=2)[1, 2]
end
function _sample_estimate(
    s::Union{Std,Covariance,Correlation,CovarianceMatrix,CorrelationMatrix},
    values,
    plan;
    joint=false,
)
    if joint && s isa Union{Covariance,Correlation}
        all(v -> length(v)==2 && all(x -> x isa Number, v), values) ||
            throw(DimensionMismatch("pairwise quantities must be scalar"))
    end
    matrix = _sample_matrix(values)
    point = _sample_point(s, matrix)
    batches = min(8, length(values)÷2)
    batches >= 2 || throw(
        ArgumentError("at least four draws are needed for a dispersion error estimate"),
    )
    width = length(values)÷batches
    points = map(1:batches) do i
        _sample_point(s, view(matrix, :, (1+(i-1)*width):(i*width)))
    end
    errors = std(_sample_matrix(points); dims=2) / sqrt(batches)
    radius = reshape(vec(errors), size(point)) .* _zscore(plan.alpha/length(point))
    if point isa Number
        radius = only(errors) * _zscore(plan.alpha)
    elseif s isa Std
        point = _shape_value(point, first(values))
        radius = _shape_value(radius, first(values))
    end
    Estimate(
        point;
        lower=point-radius,
        upper=point+radius,
        evidence=:estimated,
        n=length(values),
    )
end

function _sample_estimate(s, values, plan; joint=false)
    throw(
        ArgumentError(
            "no sampled evaluator for $s; use Derived to define a sampled quantity",
        ),
    )
end
