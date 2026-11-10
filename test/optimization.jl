using Optimization
using OptimizationOptimJL
using ForwardDiff

@testset "Optimization fitting" begin
    beliefs = (
        Approximate(:location, Mean(), 1.5; scale=1.0, importance=1.0),
        Approximate(:spread, Std(), 0.8; scale=1.0, importance=1.0),
    )
    normal = Candidate(
        :normal,
        p -> MeasureBase.AbstractMeasure(Normal(p[1], exp(p[2]))),
        ([0.0, 0.0], [2.0, -1.0]),
    )
    problem = Elicitation(beliefs, (normal,))
    report = Credence.fit(Xoshiro(42), problem, NelderMead(); maxiters=2000, g_tol=1e-12)
    @test length(report.results) == 2
    for result in report.results
        @test mean(result.measure) ≈ 1.5 atol=1e-5
        @test std(result.measure) ≈ 0.8 atol=1e-5
        @test Optimization.successful_retcode(result.solver.retcode)
    end
    @test report.results[1].id != report.results[2].id

    scaled_beliefs = map(beliefs) do belief
        Approximate(
            belief.id,
            belief.statistic,
            belief.target;
            scale=belief.scale,
            importance=1e308,
        )
    end
    scaled_report = Credence.fit(
        Xoshiro(42),
        Elicitation(scaled_beliefs, (normal,)),
        NelderMead();
        maxiters=2000,
        g_tol=1e-12,
    )
    @test mean(first(scaled_report.results).measure) ≈ 1.5 atol=1e-5
    @test std(first(scaled_report.results).measure) ≈ 0.8 atol=1e-5

    gamma = Candidate(
        :gamma,
        p -> MeasureBase.AbstractMeasure(Gamma(exp(p[1]), exp(p[2]))),
        [0.0, 0.0],
    )
    gamma_beliefs = (
        Approximate(:mean, Mean(), 6.0; scale=1.0, importance=1.0),
        Approximate(:spread, Std(), sqrt(12.0); scale=1.0, importance=1.0),
    )
    gamma_report = Credence.fit(
        Xoshiro(42),
        Elicitation(gamma_beliefs, (gamma,)),
        NelderMead();
        maxiters=2000,
        g_tol=1e-12,
    )
    parameters = only(gamma_report.results).parameters
    @test exp(parameters[1]) ≈ 3 atol=1e-4
    @test exp(parameters[2]) ≈ 2 atol=1e-4

    constrained = Elicitation(
        (
            Approximate(:location, Mean(), 2.0; scale=1.0),
            Hard(:upper_mean, Mean(), AtMost(1.0)),
        ),
        (Candidate(:normal, p -> MeasureBase.AbstractMeasure(Normal(p[1], 1.0)), [0.0]),),
    )
    constrained_report = Credence.fit(
        Xoshiro(42),
        constrained,
        IPNewton();
        adtype=Optimization.AutoForwardDiff(),
        maxiters=1000,
    )
    constrained_result = only(constrained_report.results)
    @test mean(constrained_result.measure) ≈ 1.0 atol=1e-5
    @test mean(constrained_result.measure) <= 1.0

    unfinished = Credence.fit(Xoshiro(42), problem, NelderMead(); maxiters=1)
    @test length(unfinished.results) == 2
    @test all(r -> !Optimization.successful_retcode(r.solver.retcode), unfinished.results)

    function partial_family(p)
        p[1] < 0 && throw(DomainError(p[1], "location must be nonnegative"))
        return MeasureBase.AbstractMeasure(Normal(p[1], exp(p[2])))
    end
    partial = Candidate(:partial, partial_family, ([-2.0, 0.0], [2.0, -1.0]))
    partial_report = Credence.fit(
        Xoshiro(42),
        Elicitation(beliefs, (partial,)),
        NelderMead();
        maxiters=2000,
        g_tol=1e-12,
    )
    @test partial_report.results[1].error isa DomainError
    @test mean(partial_report.results[2].measure) ≈ 1.5 atol=1e-5
end
