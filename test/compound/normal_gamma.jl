@testset "Normal–Gamma variance mixture" begin
    family = NormalGammaFamily(2; shape=0.4, scale=0.7)
    point = (coefficients=[0.2, -0.5], local_variances=[0.3, 1.8])
    prior =
        only(fit(Xoshiro(401), Elicitation((), (:theta => (:ng => family,),))).results).measure
    expected =
        sum(logpdf.(Normal.(0.0, sqrt.(point.local_variances)), point.coefficients)) +
        sum(logpdf.(Ref(Gamma(0.4, 2 * 0.7^2)), point.local_variances))
    @test logdensityof(prior, point) ≈ expected

    free = NormalGammaFamily(2; starts=(log_shape=log(0.4), log_scale=log(0.7)))
    fitted =
        only(fit(Xoshiro(402), Elicitation((), (:theta => (:ng => free,),))).results).measure
    @test logdensityof(fitted, point) ≈ expected

    laplace = NormalGammaFamily(1; shape=1.0, scale=0.7)
    laplace_prior =
        only(fit(Xoshiro(403), Elicitation((), (:theta => (:ng => laplace,),))).results).measure
    probability = evaluate(
        Probability(draw -> abs(only(draw.coefficients)) <= 0.7),
        laplace_prior;
        plan=MonteCarlo(8192),
        rng=Xoshiro(404),
    )
    @test probability.lower <= 1-exp(-1) <= probability.upper

    shape, scale = 2.0, 0.7
    moment_prior = only(
        fit(
            Xoshiro(405),
            Elicitation((), (:theta => (:ng => NormalGammaFamily(1; shape, scale),),)),
        ).results,
    ).measure
    rng = Xoshiro(407)
    draws = [only(rand(rng, moment_prior).coefficients)^2 for _ in 1:8192]
    variance = 2shape * scale^2
    error = sqrt((3 * (1 + inv(shape)) - 1) * variance^2 / length(draws))
    @test abs(mean(draws)-variance) <= 5error

    family32 = NormalGammaFamily(2; shape=0.4f0, scale=0.7f0)
    prior32 =
        only(fit(Xoshiro(406), Elicitation((), (:theta => (:ng => family32,),))).results).measure
    @test logdensityof(prior32, point) ≈ expected rtol=10eps(Float32)
end
