using VarNames: @vn

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

@testset "Sampled validation preserves undefined population moments" begin
    problem = Elicitation(
        (
            Approximate(:mean, Mean(), 0.0; scale=1.0),
            Approximate(:std, Std(), 1.0; scale=1.0),
        ),
        (
            Candidate(:cauchy, _ -> asmeasure(Cauchy()), [0.0]),
            Candidate(:normal, _ -> StdNormal(), [0.0]),
        ),
    )
    report = fit(Xoshiro(130), problem)
    validation = validate(Xoshiro(131), report; plan=MonteCarlo(128))
    @test all(check -> check.status == :undefined, validation.checks[1])
    @test all(
        check -> check.estimate.evidence == :estimated && check.estimate.n == 128,
        validation.checks[2],
    )
    @test compare(report; validation).feasible == [(:normal, 1)]
end

@testset "Correlation preserves coordinate scales" begin
    for scale in (1e200, 1e-200)
        prior = asmeasure(MvNormal(zeros(2), scale .* [1.0 0.5; 0.5 1.0]))
        @test evaluate(Correlation(), prior).value ≈ 0.5
        problem = Elicitation(
            (
                Joint(@vn(beta[1]), @vn(beta[2])) =>
                    (Hard(:correlation, Correlation(), EqualTo(0.5)),),
            ),
            (:beta => (Candidate(:normal, _ -> prior, [0.0]),),),
        )
        validation = validate(Xoshiro(132), fit(Xoshiro(133), problem))
        @test only(only(validation.checks)).estimate.value ≈ 0.5
    end
end

@testset "Independent covariance requires finite means" begin
    laws = map(
        distribution -> productmeasure([asmeasure(distribution), StdNormal()]),
        (Cauchy(), TDist(1.5)),
    )
    @test evaluate(Covariance(), laws[2]).value == 0
    for grouped in (false, true)
        local_beliefs = (
            Approximate(:covariance, Covariance(), 0.0; scale=1.0),
            Approximate(:correlation, Correlation(), 0.0; scale=1.0),
        )
        candidates = (
            Candidate(:cauchy, _ -> laws[1], [0.0]),
            Candidate(:student, _ -> laws[2], [0.0]),
        )
        if grouped
            beliefs = (Joint(@vn(beta[1]), @vn(beta[2])) => local_beliefs,)
            catalog = (:beta => candidates,)
        else
            beliefs = local_beliefs
            catalog = candidates
        end
        report = fit(Xoshiro(134), Elicitation(beliefs, catalog))
        validation = validate(Xoshiro(135), report)
        @test map(row -> map(check -> check.status, row), validation.checks) ==
              ((:undefined, :undefined), (:target, :undefined))
    end
    selected = Elicitation(
        (
            Joint(@vn(beta[1]), @vn(beta[2])) =>
                (Hard(:covariance, Covariance(), EqualTo(0.0)),),
        ),
        (
            :beta => (
                Candidate(
                    :normal_pair,
                    _ -> productmeasure([StdNormal(), StdNormal(), asmeasure(Cauchy())]),
                    [0.0],
                ),
            ),
        ),
    )
    validation = validate(Xoshiro(136), fit(Xoshiro(137), selected))
    @test only(only(validation.checks)).status == :satisfied
    pair = Joint(:a, :b)
    named = Elicitation(
        (pair => (Hard(:covariance, Covariance(), EqualTo(0.0)),),),
        (
            pair => (
                Candidate(
                    :normal_pair,
                    _ -> productmeasure((a=StdNormal(), b=StdNormal())),
                    [0.0],
                ),
            ),
        ),
    )
    @test only(only(validate(Xoshiro(138), fit(Xoshiro(139), named)).checks)).status ==
          :satisfied
    for prior in (
        productmeasure((a=asmeasure(Cauchy()), b=StdNormal())),
        productmeasure((asmeasure(Cauchy()), StdNormal())),
    )
        report = fit(
            Xoshiro(140),
            Elicitation(
                (Hard(:covariance, Covariance(), EqualTo(0.0)),),
                (Candidate(:independent, _ -> prior, [0.0]),),
            ),
        )
        @test only(only(validate(Xoshiro(141), report).checks)).status == :undefined
    end
end

@testset "Reference selections keep validation ownership" begin
    beliefs = (Hard(:mean, Mean(), Between(-0.1, 0.1)),)
    first_report = fit(
        Xoshiro(120),
        Elicitation(
            beliefs,
            (Candidate(:normal, _ -> asmeasure(Normal(10.0, 1.0)), [0.0]),),
        ),
    )
    other_report = fit(
        Xoshiro(121),
        Elicitation(beliefs, (Candidate(:normal, _ -> StdNormal(), [0.0]),)),
    )
    validation = validate(Xoshiro(122), other_report)
    @test_throws ArgumentError select(
        first_report,
        (:normal, 1);
        role=:reference,
        validation,
        rationale="Unrelated checks",
    )
    reference = select(
        first_report,
        (:normal, 1);
        role=:reference,
        rationale="Retained for discussion",
    )
    @test mean(measure(reference)) == 10.0
end
