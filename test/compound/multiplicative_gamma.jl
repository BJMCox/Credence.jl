@testset "Multiplicative Gamma loadings" begin
    family = MultiplicativeGammaFamily(
        2,
        3;
        first_shape=4.0,
        later_shape=5.0,
        local_df=8.0,
        scale=0.7,
    )
    problem = Elicitation((), (:theta => (:mgp => family,),))
    prior = only(fit(Xoshiro(1701), problem).results).measure
    point = (
        coefficients=[0.1 0.2 -0.4; -0.3 0.5 0.7],
        local_precisions=[0.8 1.1 1.3; 1.2 0.7 0.9],
        increments=[1.5, 0.4, 2.1],
    )
    column_precisions = cumprod(point.increments)
    expected =
        sum(logpdf.(Gamma(4.0, 0.25), point.local_precisions)) +
        logpdf(Gamma(4.0, 1.0), point.increments[1]) +
        sum(logpdf.(Gamma(5.0, 1.0), point.increments[2:3])) +
        sum(
            logpdf(
                Normal(
                    0.0,
                    0.7/sqrt(point.local_precisions[row, column]*column_precisions[column]),
                ),
                point.coefficients[row, column],
            ) for row in 1:2, column in 1:3
        )
    @test logdensityof(prior, point) ≈ expected

    free = MultiplicativeGammaFamily(
        2,
        3;
        first_shape=nothing,
        later_shape=nothing,
        local_df=nothing,
        scale=nothing,
        starts=(
            log_first_shape=log(4.0),
            log_later_shape=log(5.0),
            log_local_df=log(8.0),
            log_scale=log(0.7),
        ),
    )
    fitted =
        only(fit(Xoshiro(1702), Elicitation((), (:theta => (:mgp => free,),))).results).measure
    @test logdensityof(fitted, point) ≈ expected

    # Integrating the Normal and Gamma variables gives this finite variance.
    squared = Derived(:squared_loadings, @vn(theta.coefficients), x -> vec(abs2.(x)))
    variances = repeat(0.7^2 * (8/6) .* (1/3) .* (1/4) .^ (0:2); inner=2)
    beliefs = (squared => (Approximate(:mean, Mean(), variances; scale=ones(6)),),)
    report = fit(
        Xoshiro(1703),
        Elicitation(beliefs, (:theta => (:mgp => family,),));
        sampling=MonteCarlo(16384),
    )
    checks = only(validate(Xoshiro(1704), report; sampling=MonteCarlo(16384)).checks)
    estimate = only(checks).estimate
    @test all(estimate.lower .<= variances .<= estimate.upper)
    @test size(rand(Xoshiro(1705), prior).coefficients) == (2, 3)
end
