module CredenceOptimizationExt

using Credence
using MeasureBase
using HeterogeneousComputing: GenContext
using Optimization
using Random: AbstractRNG

"""
    fit(rng, problem, algorithm; adtype=Optimization.NoAD(), kwargs...)

Fit every declared candidate start with an Optimization.jl algorithm.
Hard beliefs become optimizer constraints, and each result retains solver provenance.
Analytic fits with deterministic algorithms do not consume `rng`. Configure a stochastic algorithm's own
RNG explicitly through its supported backend options; this adapter forwards `kwargs`.
"""
function Credence.fit(
    rng::Union{AbstractRNG,GenContext},
    problem::Credence.Elicitation,
    algorithm;
    adtype=Optimization.NoAD(),
    sampling=nothing,
    kwargs...,
)
    hard = filter(b -> b isa Hard, problem.beliefs)
    seed = sampling === nothing ? nothing : Credence._sampling_seed(rng)
    stream() = seed === nothing ? nothing : Credence._sampling_stream(rng, seed)
    results = map(problem.candidates) do candidate
        objective =
            (parameters, _) -> Credence._loss(
                problem,
                candidate.build(parameters);
                scope=candidate.scope,
                sampling,
                rng=stream(),
            )
        lower, upper = map((candidate.lower, candidate.upper)) do bound
            isnothing(bound) && return nothing
            collect(bound)
        end
        map(enumerate(candidate.starts)) do (index, start)
            try
                Credence._validate_candidate(candidate)
                point = copy(Credence._flatten(start))
                if isempty(point)
                    return Credence._result(
                        candidate,
                        index,
                        point,
                        candidate.build(point),
                        nothing,
                    )
                end
                initial = Credence._estimates(
                    hard,
                    candidate.build(point),
                    candidate.scope;
                    sampling,
                    rng=stream(),
                )
                layout =
                    isempty(hard) ? () :
                    Tuple(
                        Iterators.flatten(
                            map(eachindex(hard)) do i
                                estimate = initial[i]
                                estimate === nothing && return ()
                                estimate isa Credence.EvaluationFailure &&
                                    throw(estimate.error)
                                map(
                                    j -> (i, j),
                                    Credence._hard_entries(hard[i], estimate.value),
                                )
                            end,
                        ),
                    )
                constraints = if isempty(layout)
                    (;)
                else
                    bounds = map(pair -> Credence._entry_bounds(hard[pair[1]], pair[2]), layout)
                    (; lcons=collect(first.(bounds)), ucons=collect(last.(bounds)))
                end
                objective_function = if isempty(layout)
                    Optimization.OptimizationFunction(objective, adtype)
                else
                    function constraints!(values, parameters, _)
                        estimates = Credence._estimates(
                            hard,
                            candidate.build(parameters),
                            candidate.scope;
                            sampling,
                            rng=stream(),
                        )
                        for (k, (i, j)) in enumerate(layout)
                            estimate = estimates[i]
                            estimate isa Credence.EvaluationFailure &&
                                throw(estimate.error)
                            values[k] = Credence._at(estimate.value, j)
                        end
                        nothing
                    end
                    Optimization.OptimizationFunction(objective, adtype; cons=constraints!)
                end
                optimization_problem = Optimization.OptimizationProblem(
                    objective_function,
                    point;
                    lb=lower,
                    ub=upper,
                    constraints...,
                )
                solution =
                    Optimization.solve(optimization_problem, algorithm; kwargs...)
                parameters = solution.u
                return Credence._result(
                    candidate,
                    index,
                    parameters,
                    candidate.build(parameters),
                    solution,
                )
            catch error
                error isa InterruptException && rethrow()
                return Credence._failure(candidate, index, error)
            end
        end
    end
    return Credence.FitReport(problem, Tuple(Iterators.flatten(results)), sampling, seed)
end

function Credence.tilt(
    reference::MeasureBase.AbstractMeasure,
    beliefs,
    method::Credence.BoundedKL,
    algorithm;
    quadrature=Credence.Quadrature(),
    atol=1e-8,
    kwargs...,
)
    isfinite(atol) && atol > 0 || throw(ArgumentError("atol must be finite and positive"))
    beliefs = Credence._tilt_beliefs(beliefs)
    analytic = Credence._event_tilt(reference, beliefs, method)
    analytic === nothing &&
        (analytic = Credence._reference_tilt(reference, beliefs, method))
    analytic === nothing && (analytic = Credence._boundary_tilt(reference, beliefs, method))
    analytic === nothing || return analytic
    data = Credence._tilt_data(reference, beliefs, method, quadrature)
    point = zeros(eltype(data.scales), length(data.terms)+1)
    for (j, term) in enumerate(data.terms)
        if isfinite(term.lower)
            point[j+1] = one(eltype(point))
        end
    end
    objective = OptimizationFunction(
        (point, _) -> first(Credence._tilt_dual(data, point));
        grad=(gradient, point, _) ->
            copyto!(gradient, Credence._tilt_dual(data, point)[2]),
    )
    lower = [-Inf; [term.lower for term in data.terms]]
    bounds = any(isfinite, lower) ? (; lb=lower, ub=fill(Inf, length(lower))) : (;)
    problem = OptimizationProblem(objective, point; bounds...)
    solution = Optimization.solve(problem, algorithm; kwargs...)
    Credence._tilt_continuous_report(
        data,
        solution.u,
        solution;
        atol,
        converged=Optimization.successful_retcode(solution.retcode),
    )
end

include("finite_tilting.jl")

end
