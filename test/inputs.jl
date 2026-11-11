using LinearAlgebra, VarNames, HeterogeneousComputing

@testset "Named candidates and grouped judgments" begin
    beliefs = (
        :S => (
            Approximate(:median, Quantile(0.5), 1.2; scale=1.0),
            Approximate(:upper, Quantile(0.95), 3.5; scale=1.0),
        ),
    )
    candidates = (
        :S => (
            :beta=>ScaledBetaFamily(0.0, 8.0),
            :lognormal=>TruncatedFamily(LogNormal, 0.0, 8.0),
        ),
    )
    report = fit(
        Xoshiro(7),
        Elicitation(beliefs, candidates),
        NelderMead();
        maxiters=2000,
        g_tol=1e-12,
    )
    validation = validate(Xoshiro(8), report)
    @test all(r -> r.error === nothing, report.results)
    @test all(
        row ->
            isapprox(row[1].estimate.value, 1.2; atol=1e-5) &&
            isapprox(row[2].estimate.value, 3.5; atol=1e-5),
        validation.checks,
    )
    choice = select(report, :S, (:beta, 1); validation, rationale="Both quantiles match")
    @test evaluate(Probability(0.0, 8.0), measure(choice)).value ≈ 1
    custom = Candidate(
        :custom,
        p->asmeasure(Normal(p.location, exp(p.log_scale))),
        (location=2.0, log_scale=0.0),
    )
    named = fit(
        Xoshiro(9),
        Elicitation((:x=>(Approximate(:mean, Mean(), 2.0; scale=1.0),),), (:x=>(custom,),)),
    )
    @test only(named.results).parameters == (location=2.0, log_scale=0.0)
    repeated = fit(Xoshiro(9), named.problem)
    @test_throws ArgumentError select(
        repeated,
        only(repeated.results).id;
        validation=validate(Xoshiro(10), named),
        rationale="A different run",
    )
    @test_throws ArgumentError Elicitation((:x=>(), @vn(x)=>()), ())
end

@testset "Array loss, symmetry and partial bounds" begin
    family =
        Candidate(:fixed, _ -> asmeasure(MvNormal([6.0, 9.0], [4.0 1.0; 1.0 9.0])), [0.0])
    beliefs = (
        :B=>(
            Approximate(:mean, Mean(), [5.0, 7.0]; scale=[1.0, 2.0]),
            Hard(:bounds, Mean(), Between([nothing, 4.0], [8.0, missing])),
        ),
        Joint(@vn(B[1]), @vn(B[2]))=>(
            Approximate(
                :covariance,
                CovarianceMatrix(),
                UpperTriangular([4.0 1.0; 0.0 9.0]);
                scale=LowerTriangular([0.5 0.0; 0.2 1.0]),
            ),
            Hard(:correlation, Correlation(), Between(0.1, 0.2)),
        ),
    )
    report = fit(Xoshiro(1), Elicitation(beliefs, (:B=>(family,),)))
    checks = only(validate(Xoshiro(2), report).checks)
    @test checks[1].estimate.value == [6.0, 9.0]
    @test checks[2].status == :satisfied
    @test checks[3].estimate.value ≈ [4.0 1.0; 1.0 9.0]
    @test checks[4].estimate.value ≈ 1/6
    @test checks[4].status == :satisfied
    @test_throws ArgumentError Approximate(
        :bad,
        CovarianceMatrix(),
        [4.0 1.0; 2.0 9.0];
        scale=1.0,
    )
    @test_throws ArgumentError Approximate(
        :bad,
        CovarianceMatrix(),
        [4.0 1.0; 1.0 9.0];
        scale=[1.0 2.0; 3.0 1.0],
    )
    @test_throws ArgumentError Hard(
        :bad,
        CovarianceMatrix(),
        Between([0.0 nothing; 1.0 0.0], fill(10.0, 2, 2)),
    )
    @test_throws ArgumentError Approximate(
        :bad,
        CorrelationMatrix(),
        [0.0 0.3; 0.3 1.0];
        scale=1.0,
    )
    partial = Approximate(
        :partial,
        CorrelationMatrix(),
        [1.0 0.3 missing; 0.3 1.0 0.2; nothing 0.2 1.0];
        scale=0.1,
    )
    alternatives = Candidate(
        :fixed,
        p -> asmeasure(MvNormal(zeros(3), [1.0 0.3 p[1]; 0.3 1.0 0.2; p[1] 0.2 1.0])),
        ([0.0], [0.9]),
    )
    partial_report = fit(Xoshiro(3), Elicitation((:B=>(partial,),), (:B=>(alternatives,),)))
    @test compare(partial_report).ties ==
          [(partial_report.results[1].id, partial_report.results[2].id)]
    inconsistent = Approximate(
        :inconsistent,
        CorrelationMatrix(),
        [1.0 0.9 0.9; 0.9 1.0 -0.9; 0.9 -0.9 1.0];
        scale=0.1,
    )
    @test inconsistent.diagnostics == (:inconsistent_matrix_target,)
    wrong = Elicitation(
        (:B=>(Approximate(:mean, Mean(), [1.0, 2.0]; scale=1.0),),),
        (:B=>(:normal=>NormalFamily(),),),
    )
    failed = fit(Xoshiro(3), wrong, NelderMead(); maxiters=10)
    @test all(r -> r.error isa DimensionMismatch, failed.results)
    initial = fit(Xoshiro(3), wrong)
    wrong_checks = validate(Xoshiro(4), initial)
    @test all(row -> only(row).status == :undefined, wrong_checks.checks)
end

@testset "Scope composition preserves structure and bounds" begin
    direct = Elicitation(
        (
            :a => (Approximate(:mean, Mean(), 0.2; scale=1.0),),
            :b => (Approximate(:median, Quantile(0.5), -0.1; scale=1.0),),
        ),
        (
            Joint(:a, :b) => (
                Candidate(
                    :normal,
                    _ -> asmeasure(MvNormal([0.2, -0.1], [1.0 0.4; 0.4 1.0])),
                    [0.0],
                ),
            ),
        ),
    )
    checks = only(validate(Xoshiro(8), fit(Xoshiro(7), direct)).checks)
    @test checks[1].estimate.value ≈ 0.2
    @test checks[2].estimate.value ≈ -0.1
    whole = Elicitation(
        (
            Joint(:B) => (Approximate(:mean, Mean(), zeros(2); scale=1.0),),
            Derived(:dimension, :B, length) =>
                (Approximate(:mean, Mean(), 2.0; scale=1.0),),
        ),
        (
            Joint(:B) => (
                Candidate(:vector, _ -> asmeasure(MvNormal(zeros(2), ones(2))), [0.0]),
            ),
        ),
    )
    checks =
        only(validate(Xoshiro(10), fit(Xoshiro(9), whole); sampling=MonteCarlo(8)).checks)
    @test checks[1].estimate.value == zeros(2)
    @test checks[2].estimate.value == 2
    full = Candidate(
        :full,
        _ -> productmeasure((
            B=asmeasure(MvNormal(zeros(2), [1.0 0.8; 0.8 1.0])),
            C=StdNormal(),
        )),
        [0.0],
    )
    refs = Joint(@vn(B[1]), @vn(B[2]), :C)
    matrix = [1.0 0.8 0.0; 0.8 1.0 0.0; 0.0 0.0 1.0]
    joint = fit(
        Xoshiro(1),
        Elicitation(
            (refs=>(Approximate(:covariance, CovarianceMatrix(), matrix; scale=1.0),),),
            (full,),
        ),
    )
    @test only(only(validate(Xoshiro(2), joint).checks)).estimate.value == matrix
    literal = UpperTriangular([1.0 2.0; 0.0 3.0])
    alternatives = Candidate(
        :matrix,
        p -> productmeasure(map(x -> asmeasure(Normal(x, 1.0)), [1.0 2.0; p[1] 3.0])),
        ([0.0], [2.0]),
    )
    triangular = fit(
        Xoshiro(3),
        Elicitation(
            (:A=>(Approximate(:mean, Mean(), literal; scale=1.0),),),
            (:A=>(alternatives,),),
        ),
    )
    @test compare(triangular).dominates ==
          [(triangular.results[1].id, triangular.results[2].id)]
    @test_throws DimensionMismatch Elicitation(
        (Joint(:B)=>(Approximate(:pair, Covariance(), 0.0; scale=1.0),),),
        (),
    )
    @test Approximate(:negative, CovarianceMatrix(), reshape([-1e-12], 1, 1); scale=1.0).diagnostics ==
          (:inconsistent_matrix_target,)
    @test Approximate(
        :impossible,
        CorrelationMatrix(),
        [1.0 1+1e-10; 1+1e-10 1.0];
        scale=1.0,
    ).diagnostics == (:inconsistent_matrix_target,)
    named = Candidate(
        :named,
        p -> asmeasure(Normal(p.a, exp(p.b))),
        ((a=0.0, b=2.0), (a=2.0, b=2.0));
        lower=(b=-10.0, a=1.0),
    )
    bounded = fit(Xoshiro(4), Elicitation((), (named,)))
    @test bounded.results[1].error isa ArgumentError
    @test bounded.results[2].error === nothing
    catalog = (
        :x=>(
            Candidate(
                :normal,
                p -> asmeasure(Normal(p.location, 1.0)),
                ((location=0.5,), (location=2.0,));
                lower=(location=0.0,),
                upper=(location=1.0,),
            ),
        ),
    )
    composed = Elicitation((), catalog; combinations=(:joint=>Independent(:x=>:normal),))
    domain = fit(Xoshiro(5), composed)
    @test domain.results[1].error === nothing
    @test domain.results[2].error isa ArgumentError
    @test_throws ArgumentError Elicitation(
        (),
        catalog;
        combinations=(:joint=>Independent(:x=>:normal), :joint=>Independent(:x=>:normal)),
    )
    law = pushfwd(x -> (x=x, y=x), StdNormal())
    dependent = Elicitation(
        (Joint(:x, :y)=>(Hard(:equal, Probability((x, y)->x==y), AtLeast(0.9)),),),
        (Candidate(:dependent, _ -> law, [0.0]),),
    )
    checked = validate(Xoshiro(6), fit(Xoshiro(5), dependent); sampling=MonteCarlo(1000))
    @test only(only(checked.checks)).status == :satisfied
end

@testset "Shared sampling and fresh validation" begin
    counter = Ref(0)
    quantity(x) = (counter[]+=1; x^2)
    beliefs = (
        :x=>(Approximate(:mean, Mean(), 0.0; scale=1.0),),
        Derived(:square, :x, quantity)=>(
            Approximate(:mean, Mean(), 1.0; scale=1.0),
            Approximate(:upper, Quantile(0.9), 2.7; scale=1.0),
        ),
    )
    problem = Elicitation(beliefs, (:x=>(Candidate(:normal, _ -> StdNormal(), [0.0]),),))
    report = fit(Xoshiro(10), problem)
    validation = validate(Xoshiro(11), report; sampling=MonteCarlo(2000))
    checks = only(validation.checks)
    @test counter[] == 2000
    @test checks[1].estimate.n == 0
    @test checks[2].estimate.n == checks[3].estimate.n == 2000
    @test abs(checks[2].estimate.value-1) < 0.15
    @test checks[3].estimate.lower <= 2.705543454095404 <= checks[3].estimate.upper
    @test_throws ArgumentError replace(
        problem,
        Derived(:square, :x, identity) => Approximate(:mean, Mean(), 1.0; scale=1.0),
    )
    second = validate(Xoshiro(12), report; sampling=MonteCarlo(2000))
    @test validation.seed != second.seed
    @test checks[2].estimate.value != only(second.checks)[2].estimate.value
    @test_throws ArgumentError Elicitation(
        (Derived(:same, :x, identity)=>(), Derived(:same, :x, abs)=>()),
        (),
    )
end

@testset "Inactive fitting targets and caller sampling precision" begin
    problem = Elicitation(
        (
            :x => (
                Approximate(:mean, Mean(), 2.0; scale=1.0, importance=1e-100),
                Approximate(
                    :unused,
                    Statistic(_ -> error("inactive")),
                    0.0;
                    scale=1.0,
                    importance=0.0,
                ),
            ),
            :y => (Approximate(:mean, Mean(), 1.0; scale=1.0, importance=1e308),),
        ),
        (
            :x => (
                Candidate(
                    :normal,
                    p -> asmeasure(Normal(p.location, 1.0)),
                    ((location=0.0,),),
                ),
            ),
        ),
    )
    report = fit(Xoshiro(5), problem, NelderMead(); maxiters=1000, g_tol=1e-12)
    @test only(report.results).parameters.location ≈ 2.0 atol=1e-5
    types = DataType[]
    precision_problem = Elicitation(
        (
            Derived(:square, :x, x -> (push!(types, typeof(x)); x^2)) =>
                (Approximate(:mean, Mean(), 1.0; scale=1.0),),
        ),
        (:x => (Candidate(:fixed, _ -> StdNormal(), [0.0f0]),),),
    )
    checked = validate(
        GenContext{Float32}(Xoshiro(6)),
        fit(Xoshiro(7), precision_problem);
        sampling=MonteCarlo(32),
    )
    @test all(==(Float32), types)
    @test only(only(checked.checks)).estimate.n == 32
end

@testset "Sampling failures retain analytic checks" begin
    law = pushfwd(_ -> error("draw failed"), StdNormal())
    beliefs = (
        :x => (
            Approximate(:analytic, Statistic(_ -> 0.0), 0.0; scale=1.0),
            Approximate(:sampled, Probability(x -> x > 0), 0.5; scale=0.1),
        ),
    )
    report = fit(
        Xoshiro(1),
        Elicitation(beliefs, (:x => (Candidate(:fixed, _ -> law, [0.0]),),)),
    )
    checks = only(validate(Xoshiro(2), report; sampling=MonteCarlo(8)).checks)
    @test checks[1].status == :target
    @test checks[2].status == :undefined
    interrupted = Elicitation(
        (
            Approximate(
                :cancel,
                Statistic(_ -> throw(InterruptException())),
                0.0;
                scale=1.0,
            ),
        ),
        (Candidate(:normal, _ -> StdNormal(), [0.0]),),
    )
    @test_throws InterruptException validate(Xoshiro(3), fit(Xoshiro(4), interrupted))
end

@testset "Local selection and explicit independent composition" begin
    beliefs = (
        :x=>(Hard(:mean, Mean(), Between(-0.1, 0.1)),),
        :y=>(
            Approximate(:mean, Mean(), 2.0; scale=1.0),
            Hard(:ceiling, Mean(), AtMost(2.1)),
        ),
        Joint(:x, :y)=>(Hard(:correlation, Correlation(), Between(-0.01, 0.01)),),
    )
    candidates = (
        :x=>(Candidate(:normal, p->asmeasure(Normal(p.location, 1)), ((location=0.0,),)),),
        :y=>(Candidate(:normal, p->asmeasure(Normal(p.location, 1)), ((location=2.0,),)),),
    )
    problem = Elicitation(beliefs, candidates)
    report = fit(Xoshiro(1), problem)
    validation = validate(Xoshiro(2), report)
    @test validation.checks[1][3].status == :pending
    x = select(
        report,
        :x,
        (:normal, 1);
        validation,
        rationale="The x marginal meets its requirement",
    )
    joint = Elicitation(
        (beliefs[2], beliefs[3]),
        candidates;
        combinations=(:joint=>Independent(:x=>x, :y=>:normal),),
    )
    fitted = fit(
        Xoshiro(3),
        joint,
        IPNewton();
        adtype=Optimization.AutoForwardDiff(),
        maxiters=100,
    )
    check = validate(Xoshiro(4), fitted)
    @test only(fitted.results).error === nothing
    @test marginals(only(fitted.results).measure).x === measure(x)
    @test length(joint.beliefs) == 4
    accepted = select(
        fitted,
        (:joint, 1);
        validation=check,
        rationale="The joint requirements also hold",
    )
    @test mean(marginals(measure(accepted)).y) ≈ 2
    released = unlock(x)
    @test length(released.requirements) == 1
    released_joint = Elicitation(
        (:y=>(Approximate(:mean, Mean(), 5.0; scale=1.0),),),
        (unlock(accepted),),
    )
    constrained_again = fit(
        Xoshiro(8),
        released_joint,
        IPNewton();
        adtype=Optimization.AutoForwardDiff(),
        maxiters=100,
    )
    refitted_mean = mean(marginals(only(constrained_again.results).measure).y)
    @test refitted_mean ≈ 2.1 atol=1e-5
    @test refitted_mean <= 2.1
    missing_joint =
        Elicitation(beliefs, candidates; combinations=(:partial=>Independent(:x=>x),))
    partial = fit(Xoshiro(5), missing_joint)
    @test_throws ArgumentError select(
        partial,
        (:partial, 1);
        validation=validate(Xoshiro(6), partial),
        rationale="Incomplete",
    )
    exhaustive = Elicitation(beliefs, candidates; combinations=AllCombinations())
    @test length(exhaustive.candidates) == 1
end

@testset "Fixed sampling inputs and shaped array candidates" begin
    problem = Elicitation(
        (Derived(:square, :x, x->x^2)=>(Approximate(:mean, Mean(), 5.0; scale=1.0),),),
        (
            :x=>(
                Candidate(
                    :normal,
                    p->asmeasure(Normal(p.location, 1.0)),
                    ((location=1.0,),),
                ),
            ),
        ),
    )
    report = fit(
        Xoshiro(55),
        problem,
        NelderMead();
        sampling=MonteCarlo(512),
        maxiters=1000,
        g_tol=1e-12,
    )
    result = only(report.results)
    @test result.error === nothing
    rng = Xoshiro(report.seed)
    z = [rand(rng, asmeasure(Normal())) for _ in 1:512]
    root = -mean(z)+sqrt(mean(z)^2+5-mean(abs2, z))
    @test result.parameters.location ≈ root atol=1e-5
    array_problem = Elicitation(
        (:A=>(Approximate(:mean, Mean(), ones(2, 2); scale=fill(0.5, 2, 2)),),),
        (:A=>(:iid=>IIDFamily(NormalFamily(location=1.0), (2, 2)),),),
    )
    array_report = fit(Xoshiro(1), array_problem)
    @test all(
        row -> only(row).estimate.value == ones(2, 2),
        validate(Xoshiro(2), array_report).checks,
    )
    nested = Elicitation(
        (
            @vn(detector.width)=>(Approximate(:mean, Mean(), 1.0; scale=1.0),),
            Derived(:square, @vn(detector.width), x->x^2)=>(
                Approximate(:mean, Mean(), 2.0; scale=1.0),
            ),
        ),
        (
            :detector=>(
                Candidate(
                    :fixed,
                    _ -> productmeasure((width=asmeasure(Normal(1.0, 1.0)),)),
                    [0.0],
                ),
            ),
        ),
    )
    nested_report = fit(Xoshiro(3), nested)
    checks = only(validate(Xoshiro(4), nested_report; sampling=MonteCarlo(1000)).checks)
    @test checks[1].estimate.value == 1.0
    @test abs(checks[2].estimate.value-2) < 0.3
end
