module CredenceOptimizationExt

using Credence
using HeterogeneousComputing: GenContext
using Optimization
using Random: AbstractRNG

"""
    fit(rng, problem, algorithm; adtype=Optimization.NoAD(), kwargs...)

Fit every declared candidate start with an Optimization.jl algorithm.
Hard beliefs become optimizer constraints, and each result retains solver provenance.
Deterministic algorithms do not consume `rng`. Configure a stochastic algorithm's own
RNG explicitly through its supported backend options; this adapter forwards `kwargs`.
"""
function Credence.fit(
    rng::Union{AbstractRNG,GenContext},
    problem::Credence.Elicitation,
    algorithm;
    adtype=Optimization.NoAD(),
    kwargs...,
)
    hard = filter(b -> b isa Hard, problem.beliefs)
    constraints = if isempty(hard)
        (;)
    else
        bounds = map(b -> Credence._bounds(b.relation), hard)
        (; lcons=first.(bounds), ucons=last.(bounds))
    end
    results = map(problem.candidates) do candidate
        objective = (parameters, _) -> Credence._loss(problem, candidate.build(parameters))
        objective_function = if isempty(hard)
            Optimization.OptimizationFunction(objective, adtype)
        else
            function constraints!(values, parameters, _)
                prior = candidate.build(parameters)
                values .= map(b -> evaluate(b.statistic, prior).value, hard)
                return nothing
            end
            Optimization.OptimizationFunction(objective, adtype; cons=constraints!)
        end
        lower, upper = map((candidate.lower, candidate.upper)) do bound
            isnothing(bound) && return nothing
            collect(bound)
        end
        map(enumerate(candidate.starts)) do (index, start)
            try
                optimization_problem = Optimization.OptimizationProblem(
                    objective_function,
                    collect(start);
                    lb=lower,
                    ub=upper,
                    map(collect, constraints)...,
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
    return Credence.FitReport(problem, Tuple(Iterators.flatten(results)))
end

end
