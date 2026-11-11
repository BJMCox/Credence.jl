@testset "Dirichlet–Laplace hierarchy" begin
    family = DirichletLaplaceFamily(3; concentration=0.5, scale=0.4)
    problem = Elicitation((), (:theta => (:dl => family,),))
    report = fit(Xoshiro(1301), problem)
    prior = only(report.results).measure
    point = (coefficients=[0.1, -0.2, 0.4], global_scale=1.7, allocation=[0.2, 0.3, 0.5])
    expected =
        logpdf(Gamma(1.5, 2.0), point.global_scale) +
        logpdf(Dirichlet(fill(0.5, 3)), point.allocation) +
        sum(
            logpdf(Laplace(0.0, 0.4*point.global_scale*weight), coefficient) for
            (weight, coefficient) in zip(point.allocation, point.coefficients)
        )
    @test logdensityof(prior, point) ≈ expected

    family = DirichletLaplaceFamily(
        3;
        concentration=nothing,
        scale=nothing,
        starts=(log_concentration=log(0.5), log_scale=log(0.4)),
    )
    report = fit(Xoshiro(1302), Elicitation((), (:theta => (:dl => family,),)))
    @test logdensityof(only(report.results).measure, point) ≈ expected

    threshold = quantile(Gamma(0.5, 2.0), 0.5)
    marginal = evaluate(
        Probability(x -> x.global_scale*x.allocation[1] <= threshold),
        prior;
        plan=MonteCarlo(8192),
        rng=Xoshiro(1303),
    )
    joint = evaluate(
        Probability(x -> all(x.global_scale .* x.allocation[1:2] .<= threshold)),
        prior;
        plan=MonteCarlo(8192),
        rng=Xoshiro(1304),
    )
    @test marginal.lower <= 0.5 <= marginal.upper
    @test joint.lower <= 0.25 <= joint.upper
end
