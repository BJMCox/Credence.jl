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
    lower_constraints, upper_constraints = Credence._hard_bounds(problem)
    results = map(problem.candidates) do candidate
        objective = (parameters, _) -> Credence._loss(problem, candidate.build(parameters))
        objective_function = if isempty(lower_constraints)
            Optimization.OptimizationFunction(objective, adtype)
        else
            function constraints!(values, parameters, _)
                copyto!(values, Credence._hard_values(problem, candidate.build(parameters)))
                return nothing
            end
            Optimization.OptimizationFunction(objective, adtype; cons=constraints!)
        end
        lower = candidate.lower
        if !isnothing(lower)
            lower = collect(lower)
        end
        upper = candidate.upper
        if !isnothing(upper)
            upper = collect(upper)
        end
        map(enumerate(candidate.starts)) do (index, start)
            try
                optimization_problem = if isempty(lower_constraints)
                    Optimization.OptimizationProblem(
                        objective_function,
                        collect(start);
                        lb=lower,
                        ub=upper,
                    )
                else
                    Optimization.OptimizationProblem(
                        objective_function,
                        collect(start);
                        lb=lower,
                        ub=upper,
                        lcons=collect(lower_constraints),
                        ucons=collect(upper_constraints),
                    )
                end
                solution =
                    Optimization.solve(optimization_problem, algorithm; kwargs...)
                parameters = solution.u
                return Credence._result(
                    problem,
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
