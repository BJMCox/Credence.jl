@testset "Generalized double Pareto hierarchy" begin
    family = GeneralizedDoubleParetoFamily(2; shape=3.0, scale=0.4)
    prior =
        only(fit(Xoshiro(1901), Elicitation((), (:theta => (:gdp => family,),))).results).measure
    point = (coefficients=[0.2, -0.7], local_rates=[0.6, 2.1])
    expected =
        sum(logpdf.(Ref(Gamma(3.0, 1.0)), point.local_rates)) + sum(
            logpdf(Laplace(0.0, 1.2/rate), coefficient) for
            (rate, coefficient) in zip(point.local_rates, point.coefficients)
        )
    @test logdensityof(prior, point) ≈ expected

    fitted = GeneralizedDoubleParetoFamily(
        2;
        shape=nothing,
        starts=(log_shape=log(3.0), log_scale=log(0.4)),
    )
    fitted_prior =
        only(fit(Xoshiro(1902), Elicitation((), (:theta => (:gdp => fitted,),))).results).measure
    @test logdensityof(fitted_prior, point) ≈ expected

    heavy = GeneralizedDoubleParetoFamily(2; shape=1.0, scale=0.4)
    heavy_prior =
        only(fit(Xoshiro(1903), Elicitation((), (:theta => (:gdp => heavy,),))).results).measure
    # The exact marginal CDF gives P(X <= 3ξ)=7/8 for shape one.
    marginal = evaluate(
        Probability(point -> point.coefficients[1] <= 1.2),
        heavy_prior;
        plan=MonteCarlo(8192),
        rng=Xoshiro(1904),
    )
    joint = evaluate(
        Probability(point -> all(point.coefficients .<= 1.2)),
        heavy_prior;
        plan=MonteCarlo(8192),
        rng=Xoshiro(1905),
    )
    @test marginal.lower <= 7/8 <= marginal.upper
    @test joint.lower <= (7/8)^2 <= joint.upper
end
