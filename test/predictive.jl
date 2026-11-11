@testset "Predictive beliefs share observations" begin
    calls = Ref(0)
    kernel = mu -> begin
        calls[] += 1
        asmeasure(MvNormal([mu, 2mu], [0.25 0.0; 0.0 0.25]))
    end
    observable = Credence.Predictive(:measurement, :mu, kernel)
    covariance = [1.25 2.0; 2.0 4.25]
    beliefs = (
        observable => (
            Approximate(:mean, Mean(), [2.0, 4.0]; scale=1.0),
            Approximate(:covariance, CovarianceMatrix(), covariance; scale=1.0),
            Approximate(:positive, Probability(x -> x[2] > 4), 0.5; scale=1.0),
            Approximate(:pair, Covariance(), 2.0; scale=1.0),
        ),
    )
    candidates = (:mu => (Candidate(:normal, _ -> asmeasure(Normal(2.0, 1.0)), [0.0]),),)
    problem = Elicitation(beliefs, candidates)
    n = 4096
    validation = validate(Xoshiro(21), fit(Xoshiro(20), problem); sampling=MonteCarlo(n))
    checks = only(validation.checks)
    @test calls[] == n
    @test all(abs.(checks[1].estimate.value - [2.0, 4.0]) .<= 6sqrt.(diag(covariance) / n))
    covariance_errors =
        sqrt.((diag(covariance) * diag(covariance)' + covariance .^ 2) / (n-1))
    @test all(abs.(checks[2].estimate.value - covariance) .<= 6covariance_errors)
    probability = checks[3].estimate
    @test probability.lower <= 0.5 <= probability.upper
    @test checks[4].estimate.value == checks[2].estimate.value[1, 2]

    failed_calls = Ref(0)
    failed_kernel = mu -> begin
        failed_calls[] += 1
        error("observation model failed")
    end
    failed = Credence.Predictive(:failed_observation, :mu, failed_kernel)
    failed_beliefs = (
        failed => (
            Approximate(:mean, Mean(), 2.0; scale=1.0),
            Approximate(:median, Quantile(0.5), 2.0; scale=1.0),
        ),
    )
    validation = validate(
        Xoshiro(23),
        fit(Xoshiro(22), Elicitation(failed_beliefs, candidates));
        sampling=MonteCarlo(16),
    )
    @test failed_calls[] == 1
    @test all(
        check -> check.status == :undefined && check.error !== nothing,
        only(validation.checks),
    )
end

@testset "Gamma-Poisson predictive checks" begin
    marginal = NegativeBinomial(2.0, 1/7)
    median = quantile(marginal, 0.5)
    probability = cdf(marginal, 20) - cdf(marginal, 0)
    observable = Credence.Predictive(:count, :rate, rate -> asmeasure(Poisson(2rate)))
    beliefs = (
        observable => (
            Approximate(:mean, Mean(), 12.0; scale=1.0),
            Approximate(:median, Quantile(0.5), median; scale=1.0),
            Approximate(:event, Probability(0.0, 20.0), probability; scale=1.0),
        ),
    )
    candidates = (:rate => (Candidate(:gamma, _ -> asmeasure(Gamma(2.0, 3.0)), [0.0]),),)
    n = 4096
    report = fit(Xoshiro(30), Elicitation(beliefs, candidates); sampling=MonteCarlo(n))
    validation = validate(Xoshiro(31), report)
    checks = only(validation.checks)
    @test abs(checks[1].estimate.value - 12) <= 6sqrt(var(marginal) / n)
    @test checks[2].estimate.lower <= median <= checks[2].estimate.upper
    @test checks[3].estimate.lower <= probability <= checks[3].estimate.upper
    fresh = validate(Xoshiro(32), report)
    @test fresh.seed != validation.seed
    @test only(fresh.checks)[1].estimate.value != checks[1].estimate.value
end

@testset "Predictive fitting replays random inputs" begin
    observable = Credence.Predictive(:observation, :mu, mu -> asmeasure(Normal(mu, 1.0)))
    beliefs = (observable => (Approximate(:mean, Mean(), 3.0; scale=1.0),),)
    candidate = Candidate(:normal, p -> asmeasure(Normal(p.location, 0.5)), (location=0.0,))
    problem = Elicitation(beliefs, (:mu => (candidate,),))
    n = 256
    first_report =
        fit(Xoshiro(40), problem, NelderMead(); sampling=MonteCarlo(n), maxiters=500)
    repeated_report =
        fit(Xoshiro(40), problem, NelderMead(); sampling=MonteCarlo(n), maxiters=500)
    @test only(first_report.results).parameters == only(repeated_report.results).parameters
    validation = validate(Xoshiro(41), first_report; sampling=MonteCarlo(4096))
    achieved = only(only(validation.checks)).estimate
    @test first_report.seed != validation.seed
    @test abs(achieved.value - 3) <= 6sqrt(1.25/n)
    selection = select(
        first_report,
        :mu,
        (:normal, 1);
        validation,
        rationale="Predictive mean reviewed",
    )
    @test abs(mean(measure(selection)) - 3) <= 6sqrt(1.25/n)
end

@testset "Predictive simulators use the explicit stream" begin
    calls = Ref(0)
    simulator = (rng, mu) -> begin
        calls[] += 1
        [mu + rand(rng, Normal()), mu + rand(rng, Normal())]
    end
    observable = Credence.Predictive(:readings, :mu; simulator)
    beliefs = (
        observable => (
            Approximate(:mean, Mean(), [2.0, 2.0]; scale=1.0),
            Approximate(:event, Probability(x -> x[1] > x[2]), 0.5; scale=1.0),
        ),
    )
    candidate = Candidate(:fixed, _ -> MeasureBase.Dirac(2.0), [0.0])
    report = fit(
        Xoshiro(50),
        Elicitation(beliefs, (:mu => (candidate,),));
        sampling=MonteCarlo(256),
    )
    first_validation = validate(Xoshiro(51), report)
    @test calls[] == 256
    repeated_validation = validate(Xoshiro(51), report)
    @test first_validation.checks[1][1].estimate.value ==
          repeated_validation.checks[1][1].estimate.value
    @test first_validation.checks[1][2].estimate.value ==
          repeated_validation.checks[1][2].estimate.value
    probability = first_validation.checks[1][2].estimate
    @test probability.lower <= 0.5 <= probability.upper
end
