@testset "Diagnostics preserve conflicting judgments" begin
    quantiles = (
        :x => (
            Approximate(:lower, Quantile(0.25), [2.0, missing]; scale=1.0),
            Hard(:upper, Quantile(0.75), AtMost([1.0, 3.0])),
        ),
        :y => (Approximate(:lower, Quantile(0.25), -10.0; scale=1.0),),
    )
    problem = Elicitation(quantiles, ())
    conflicts = diagnostics(problem)
    @test only(conflicts).code == :inconsistent_quantiles
    @test only(conflicts).id == (problem.beliefs[1].id, problem.beliefs[2].id)
    @test problem[:x, :lower].target[1] == 2.0
    revised = replace(
        problem,
        :x => Approximate(:lower, Quantile(0.25), [0.0, missing]; scale=1.0),
    )
    @test isempty(diagnostics(revised))

    events = Elicitation(
        (
            Approximate(:inner, Probability(0.0, 1.0), 0.8; scale=1.0),
            Hard(:outer, Probability(-1.0, 2.0), AtMost(0.4)),
            Approximate(:disjoint, Probability(3.0, 4.0), 0.1; scale=1.0),
        ),
        (),
    )
    @test only(diagnostics(events)).code == :inconsistent_probabilities
    consistent = Elicitation(
        (
            Approximate(:inner, Probability(0.0, 1.0), 0.3; scale=1.0),
            Hard(:outer, Probability(-1.0, 2.0), AtMost(0.4)),
        ),
        (),
    )
    @test isempty(diagnostics(consistent))
end
