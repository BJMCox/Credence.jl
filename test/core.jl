@testset "Analytic statistics and explicit selection" begin
    normal = asmeasure(Normal(1, 2))
    @test evaluate(Quantile(0.5), normal).value ≈ 1
    @test evaluate(Probability(-Inf, 1), normal).value ≈ 0.5
    @test evaluate(Mean(), normal).value ≈ 1
    @test evaluate(Std(), normal).value ≈ 2
    beliefs = (Hard(:center, Quantile(0.5), Between(0.9, 1.1)),)
    candidate = Candidate(:normal, p -> asmeasure(Normal(p[1], 2)), ([1.0], [3.0]))
    report = fit(Xoshiro(1), Elicitation(beliefs, (candidate,)))
    checks = validate(Xoshiro(2), report)
    @test checks.checks[1][1].status == :satisfied
    @test checks.checks[2][1].status == :violated
    choice = select(
        report,
        (:normal, 1);
        validation=checks,
        rationale="Median matches the stated range",
    )
    @test measure(choice) === report.results[1].measure
    @test_throws ArgumentError select(
        report,
        (:normal, 2);
        validation=checks,
        rationale="Invalid",
    )
    @test compare(report).feasible == [(:normal, 1)]
end

@testset "Partial comparison preserves tradeoffs and alternatives" begin
    beliefs = (
        Approximate(:location, Mean(), 0.0; scale=1.0),
        Approximate(:spread, Std(), 2.0; scale=1.0),
    )
    candidate = Candidate(
        :normal,
        p -> asmeasure(Normal(p...)),
        ([0.0, 1.0], [1.0, 2.0], [0.0, 1.0], [2.0, 3.0]),
    )
    report = fit(Xoshiro(1), Elicitation(beliefs, (candidate,)))
    comparison = compare(report)
    @test ((:normal, 1), (:normal, 2)) in comparison.incomparable
    @test ((:normal, 1), (:normal, 3)) in comparison.ties
    @test ((:normal, 2), (:normal, 4)) in comparison.dominates
    @test length(report.results) == 4
    @test all(isempty, compare(fit(Xoshiro(1), Elicitation((), ()))))
end

@testset "Failures and uncertainty remain visible" begin
    estimated = Statistic(_ -> Estimate(0.5; evidence=:estimated))
    uncertain = Statistic(_ -> Estimate(0.5; lower=0.4, upper=0.6, evidence=:statistical))
    problem = Elicitation(
        (Hard(:event, estimated, EqualTo(0.5)),),
        (
            Candidate(:valid, _ -> StdNormal(), [0.0]),
            Candidate(:broken, _ -> error("model failure"), [0.0]),
        ),
    )
    report = fit(Xoshiro(1), problem)
    checks = validate(Xoshiro(2), report)
    @test report.results[2].error isa ErrorException
    @test checks.checks[1][1].status == :inconclusive
    @test checks.checks[2][1].status == :undefined
    diagnostic = fit(
        Xoshiro(1),
        Elicitation((Hard(:event, estimated, EqualTo(0.1)),), problem.candidates[1:1]),
    )
    @test only(only(validate(Xoshiro(2), diagnostic).checks)).status == :inconclusive
    @test_throws ArgumentError select(
        report,
        (:valid, 1);
        validation=checks,
        rationale="Unresolved",
    )
    soft = Elicitation(
        (Approximate(:event, uncertain, 0.5; scale=1),),
        problem.candidates[1:1],
    )
    rsoft = fit(
        Xoshiro(1),
        Elicitation(
            soft.beliefs,
            (
                Candidate(:a, _ -> StdNormal(), [0.0]),
                Candidate(:b, _ -> StdNormal(), [0.0]),
            ),
        ),
    )
    @test compare(rsoft).unresolved == [((:a, 1), (:b, 1))]
    diagnostic_soft = fit(
        Xoshiro(1),
        Elicitation(
            (Approximate(:event, estimated, 0.5; scale=1),),
            rsoft.problem.candidates,
        ),
    )
    @test compare(diagnostic_soft).unresolved == [((:a, 1), (:b, 1))]
end

@testset "Sampling validation records uncertainty" begin
    problem = Elicitation(
        (Hard(:lower_half, Probability(-Inf, 0), AtLeast(0.4)),),
        (Candidate(:normal, _ -> StdNormal(), [0.0]),),
    )
    report = fit(Xoshiro(1), problem)
    checks = validate(Xoshiro(2), report; plan=MonteCarlo(10_000; alpha=0.01))
    check = only(only(checks.checks))
    @test check.status == :satisfied
    @test check.estimate.lower <= 0.5 <= check.estimate.upper
    @test check.estimate.evidence == :statistical
end

@testset "Parameter domain and point-mass semantics" begin
    @test evaluate(Quantile(0.5), MeasureBase.Dirac(2)).value == 2
    @test evaluate(Probability(0, 2), MeasureBase.Dirac(2)).value == 1
    @test evaluate(Probability(2, 3), MeasureBase.Dirac(2)).value == 0
    candidate =
        Candidate(:bounded, _ -> StdNormal(), ([0.0], [2.0]); lower=[-1.0], upper=[1.0])
    report = fit(Xoshiro(1), Elicitation((), (candidate,)))
    @test report.results[1].error === nothing
    @test report.results[2].error isa ArgumentError
end
