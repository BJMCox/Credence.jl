@testset "Positive and heavy-tailed families" begin
    beliefs = (
        :x => (
            Approximate(:mean, Mean(), 6.0; scale=1.0),
            Approximate(:std, Std(), sqrt(18.0); scale=1.0),
        ),
    )
    problem = Elicitation(beliefs, (:x => (:gamma => Credence.GammaFamily(),),))
    report = fit(Xoshiro(801), problem, NelderMead(); maxiters=2000, g_tol=1e-12)
    prior = only(report.results).measure
    @test mean(prior) ≈ 6.0 atol=1e-5
    @test std(prior) ≈ sqrt(18.0) atol=1e-5

    problem = Elicitation(
        (:x => (Approximate(:upper, Quantile(0.75), 5.0; scale=1.0),),),
        (:x => (:student => Credence.StudentTFamily(1; location=2.0),),),
    )
    report = fit(Xoshiro(802), problem, NelderMead(); maxiters=1000, g_tol=1e-12)
    prior = only(report.results).measure
    @test evaluate(Quantile(0.75), prior).value ≈ 5.0 atol=1e-5
    @test !isfinite(mean(prior))

    problem = Elicitation(
        (:x => (Approximate(:median, Quantile(0.5), 2.0; scale=1.0),),),
        (:x => (:normal => NormalFamily(scale=3.0),),),
    )
    report = fit(Xoshiro(815), problem, NelderMead(); maxiters=1000, g_tol=1e-12)
    @test all(
        result ->
            isapprox(mean(result.measure), 2.0; atol=1e-5) && std(result.measure) == 3.0,
        report.results,
    )
    fixed = LogNormalFamily(location=0.3, scale=0.7)
    report = fit(Xoshiro(816), Elicitation((), (:x => (:lognormal => fixed,),)))
    @test only(report.results).parameters == NamedTuple()
    @test logdensityof(only(report.results).measure, 1.0) ≈ logpdf(LogNormal(0.3, 0.7), 1.0)
    bounded = TruncatedFamily(NormalFamily(scale=2.0), -3.0, 3.0)
    report = fit(Xoshiro(817), Elicitation((), (:x => (:bounded => bounded,),)))
    prior = only(report.results).measure
    expected = truncated(Normal(0.0, 2.0), -3.0, 3.0)
    @test logdensityof(prior, 1.0) ≈ logpdf(expected, 1.0)
    @test logdensityof(prior, 4.0) == -Inf
    @test massof(prior) == 1
    @test evaluate(Probability(-3.0, 3.0), prior).value ≈ 1
    @test evaluate(Mean(), prior).value ≈ mean(expected)
    @test evaluate(Std(), prior).value ≈ std(expected)

    function truncated_median(location)
        family = TruncatedFamily(LogNormalFamily(; location, scale=0.5), 0.0, Inf)
        report = fit(Xoshiro(818), Elicitation((), (:x => (:law => family,),)))
        evaluate(Quantile(0.5), only(report.results).measure).value
    end
    @test truncated_median(0.0) ≈ 1
    @test ForwardDiff.derivative(truncated_median, 0.0) ≈ 1

    cases = (
        (Credence.WeibullFamily(shape=2.0, scale=3.0), Weibull(2.0, 3.0)),
        (Credence.ExponentialFamily(scale=3.0), Exponential(3.0)),
        (Credence.CauchyFamily(location=2.0, scale=3.0), Cauchy(2.0, 3.0)),
        (Credence.LaplaceFamily(location=2.0, scale=3.0), Laplace(2.0, 3.0)),
        (Credence.HalfNormalFamily(scale=2.0), truncated(Normal(0.0, 2.0), 0.0, Inf)),
        (
            Credence.HalfStudentTFamily(1; scale=3.0),
            truncated(LocationScale(0.0, 3.0, TDist(1)), 0.0, Inf),
        ),
    )
    for (family, expected) in cases
        problem = Elicitation((), (:x => (:law => family,),))
        prior = only(fit(Xoshiro(803), problem).results).measure
        @test logdensityof(prior, 1.0) ≈ logpdf(expected, 1.0)
    end
end

@testset "Dirichlet families and analytic marginals" begin
    expected = Dirichlet([2.0, 3.0, 5.0])
    problem = Elicitation(
        (
            :weights => (
                Approximate(:mean, Mean(), mean(expected); scale=0.1),
                Approximate(:covariance, CovarianceMatrix(), cov(expected); scale=0.01),
            ),
        ),
        (:weights => (:dirichlet => Credence.DirichletFamily(3; mean=mean(expected)),),),
    )
    report = fit(Xoshiro(804), problem, NelderMead(); maxiters=2000, g_tol=1e-12)
    prior = only(report.results).measure
    @test mean(prior) ≈ mean(expected)
    @test Credence.evaluate(CovarianceMatrix(), prior).value ≈ cov(expected) atol=1e-6

    beliefs = (
        @vn(weights[1]) => (
            Approximate(:median, Quantile(0.5), quantile(Beta(2.0, 8.0), 0.5); scale=0.1),
        ),
        @vn(weights[1:2]) => (
            Approximate(:mean, Mean(), mean(expected)[1:2]; scale=0.1),
            Approximate(
                :covariance,
                CovarianceMatrix(),
                cov(expected)[1:2, 1:2];
                scale=0.01,
            ),
        ),
    )
    exact = Credence.DirichletFamily(3; starts=(log_alpha=log.([2.0, 3.0, 5.0]),))
    validation = validate(
        Xoshiro(806),
        fit(Xoshiro(805), Elicitation(beliefs, (:weights => (:law => exact,),))),
    )
    @test all(
        check -> check.status == :target && check.estimate.evidence == :analytic,
        only(validation.checks),
    )
    @test only(validation.checks)[2].estimate.value ≈ [0.2, 0.3]
    @test only(validation.checks)[3].estimate.value[1, 2] < 0

    fixed_total =
        Credence.DirichletFamily(3; concentration=10.0, starts=(logits=log.([0.4, 0.6]),))
    prior = only(
        fit(Xoshiro(807), Elicitation((), (:weights => (:law => fixed_total,),))).results,
    ).measure
    @test mean(prior) ≈ [0.2, 0.3, 0.5]
    @test cov(prior.obj) ≈ cov(expected)
    @test evaluate(Quantile(0.5), prior).value ≈
          [quantile(Beta(alpha, 10.0-alpha), 0.5) for alpha in (2.0, 3.0, 5.0)]
end

@testset "Multivariate Student-t scatter and marginals" begin
    point = (location=[1.0, -2.0], log_scales=log.([2.0, 1.0]), lower=[0.4])
    family = Credence.MvStudentTFamily(2; df=4.0, starts=point)
    scatter = [4.0 0.8; 0.8 1.16]
    problem = Elicitation(
        (
            :beta => (
                Approximate(:covariance, CovarianceMatrix(), 2 .* scatter; scale=1.0),
            ),
            @vn(beta[1]) => (
                Approximate(
                    :upper,
                    Quantile(0.75),
                    1.0 + 2quantile(TDist(4), 0.75);
                    scale=1.0,
                ),
            ),
            @vn(beta[1:2]) => (Approximate(:mean, Mean(), [1.0, -2.0]; scale=1.0),),
        ),
        (:beta => (:student => family,),),
    )
    validation = validate(Xoshiro(809), fit(Xoshiro(808), problem))
    @test all(
        check -> check.status == :target && check.estimate.evidence == :analytic,
        only(validation.checks),
    )
    @test only(validation.checks)[1].estimate.value ≈ 2 .* scatter
    @test only(validation.checks)[2].estimate.value ≈ 1.0 + 2quantile(TDist(4), 0.75)

    free_df = Credence.MvStudentTFamily(
        2;
        starts=(log_df=log(0.8), location=[1.0, -2.0], log_scales=zeros(2), lower=[0.0]),
    )
    problem = Elicitation(
        (@vn(beta[1]) => (Approximate(:median, Quantile(0.5), 1.0; scale=1.0),),),
        (:beta => (:student => free_df,),),
    )
    validation = validate(Xoshiro(811), fit(Xoshiro(810), problem))
    @test only(only(validation.checks)).status == :target
    @test only(only(validation.checks)).estimate.value ≈ 1.0

    joint = Joint(:a, :b)
    problem = Elicitation(
        (
            :a => (Approximate(:median, Quantile(0.5), 1.0; scale=1.0),),
            :b => (Approximate(:median, Quantile(0.5), -2.0; scale=1.0),),
        ),
        (joint => (:student => free_df,),);
        combinations=(:joint => Independent(joint => :student),),
    )
    report = fit(Xoshiro(812), problem)
    validation = validate(Xoshiro(813), report)
    prior = measure(select(report, (:joint, 1); validation, rationale="Quantile judgments"))
    draw = rand(Xoshiro(814), prior)
    @test all(isfinite, (draw.a, draw.b))
    @test logdensityof(prior, (a=1.3, b=-1.6)) ≈
          logpdf(MvTDist(0.8, [1.0, -2.0], Matrix{Float64}(I, 2, 2)), [1.3, -1.6])
end
