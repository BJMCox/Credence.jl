@testset "R2D2 variance allocation and joint density" begin
    family = R2D2Family(
        3;
        noise_scale=2.0,
        predictor_scales=[1.0, 2.0, 3.0],
        mean_r2=0.25,
        precision_r2=16.0,
        concentration=0.8,
    )
    report = fit(Xoshiro(1301), Elicitation((), (:theta => (:r2d2 => family,),)))
    prior = only(report.results).measure
    point = (coefficients=[0.3, -0.2, 0.1], r2=0.4, allocation=[0.2, 0.3, 0.5])
    scales = 2 .* sqrt.(point.allocation .* (0.4/(2*0.6))) ./ [1, 2, 3]
    expected =
        logpdf(Beta(4.0, 12.0), point.r2) +
        logpdf(Dirichlet(fill(0.8, 3)), point.allocation) +
        sum(
            logpdf(Laplace(0.0, scale), beta) for
            (scale, beta) in zip(scales, point.coefficients)
        )
    @test logdensityof(prior, point) ≈ expected

    rng = Xoshiro(1302)
    n = 8192
    draws = [rand(rng, prior) for _ in 1:n]
    @test all(
        draw -> length(draw.coefficients) == 3 && isapprox(sum(draw.allocation), 1),
        draws,
    )
    r2_mean = 0.25
    r2_variance = 0.25 * 0.75 / 17
    @test abs(mean(draw.r2 for draw in draws) - r2_mean) < 6sqrt(r2_variance/n)

    # The conditional coefficient variances sum to the supplied variance budget.
    budgets = [sum(abs2, draw.coefficients .* [1, 2, 3]) / 4 for draw in draws]
    expected_budget = 4/11
    second_budget = 4*5/(11*10)
    expected_sum_squares = (0.8+1)/(3*0.8+1)
    variance = second_budget * (1 + 5expected_sum_squares) - expected_budget^2
    @test abs(mean(budgets)-expected_budget) < 6sqrt(variance/n)
end

@testset "R2D2 free and fixed elicitation parameters" begin
    family = R2D2Family(
        2;
        noise_scale=1.0,
        predictor_scales=ones(2),
        concentration=nothing,
        starts=(
            logit_mean_r2=log(1/3),
            log_precision_r2=log(8.0),
            log_concentration=log(0.5),
        ),
    )
    report = fit(Xoshiro(1303), Elicitation((), (:theta => (:r2d2 => family,),)))
    prior = only(report.results).measure
    point = (coefficients=[0.3, -0.4], r2=0.6, allocation=[0.25, 0.75])
    scales = sqrt.(point.allocation .* (0.6/(2*0.4)))
    expected =
        logpdf(Beta(2.0, 6.0), 0.6) +
        logpdf(Dirichlet([0.5, 0.5]), point.allocation) +
        sum(
            logpdf(Laplace(0.0, scale), beta) for
            (scale, beta) in zip(scales, point.coefficients)
        )
    @test logdensityof(prior, point) ≈ expected
    @test only(report.results).parameters == family.starts
end
