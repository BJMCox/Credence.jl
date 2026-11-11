@testset "Functional horseshoe subspace law" begin
    design = [1.0 -1.0 1.0; 1.0 0.0 0.0; 1.0 1.0 1.0; 1.0 2.0 4.0]
    baseline = design[:, 1:2]
    family = FunctionalHorseshoeFamily(design, baseline; baseline_scale=3.0, scale=0.4)
    prior =
        only(fit(Xoshiro(1701), Elicitation((), (:theta=>(:fhs=>family,),))).results).measure
    point = (coefficients=[0.3, -0.2, 0.1], shrinkage=0.8)
    projection = baseline * inv(baseline' * baseline) * baseline'
    gram = design' * design
    penalty = design' * (I - projection) * design
    precision = (gram - penalty) / 3.0^2 + penalty / (0.4^2 * 0.25)
    covariance = inv(Symmetric(precision))
    expected =
        logpdf(Beta(0.5, 0.5), 0.8) +
        logpdf(MvNormal(zeros(3), Symmetric(covariance)), point.coefficients)
    @test logdensityof(prior, point) ≈ expected

    equivalent = FunctionalHorseshoeFamily(
        design,
        hcat(2baseline[:, 2], -baseline[:, 1], sum(baseline; dims=2));
        baseline_scale=3.0,
        scale=0.4,
    )
    other =
        only(fit(Xoshiro(1702), Elicitation((), (:theta=>(:fhs=>equivalent,),))).results).measure
    @test logdensityof(other, point) ≈ expected

    null = FunctionalHorseshoeFamily(design, zeros(4, 0); baseline_scale=3.0, scale=0.4)
    null_prior =
        only(fit(Xoshiro(1703), Elicitation((), (:theta=>(:fhs=>null,),))).results).measure
    expected_null =
        logpdf(Beta(0.5, 0.5), 0.8) +
        logpdf(MvNormal(zeros(3), Symmetric(0.04inv(gram))), point.coefficients)
    @test logdensityof(null_prior, point) ≈ expected_null

    free = FunctionalHorseshoeFamily(
        design,
        baseline;
        baseline_scale=3.0,
        starts=(log_scale=log(0.4),),
    )
    free_prior =
        only(fit(Xoshiro(1704), Elicitation((), (:theta=>(:fhs=>free,),))).results).measure
    @test logdensityof(free_prior, point) ≈ expected

    @test_throws ArgumentError FunctionalHorseshoeFamily(
        hcat(design, design[:, 1]),
        baseline;
        baseline_scale=3.0,
    )
    @test_throws ArgumentError FunctionalHorseshoeFamily(
        design,
        [0.0, 0.0, 0.0, 1.0];
        baseline_scale=3.0,
    )
    @test_throws ArgumentError FunctionalHorseshoeFamily(design, design; baseline_scale=3.0)
end

@testset "Functional horseshoe geometry remains tied to validation" begin
    design = [1.0 -1.0 1.0; 1.0 0.0 0.0; 1.0 1.0 1.0; 1.0 2.0 4.0]
    family =
        FunctionalHorseshoeFamily(design, design[:, 1:2]; baseline_scale=3.0, scale=0.4)
    problem = Elicitation((), (:theta=>(:fhs=>family,),))
    report = fit(Xoshiro(1707), problem)
    validation = validate(Xoshiro(1708), report)
    family.design[1, 3] += 0.5
    @test only(fit(Xoshiro(1709), problem).results).error isa ArgumentError
    mktempdir() do directory
        @test_throws ArgumentError save_session(
            joinpath(directory, "changed-geometry.toml"),
            problem;
            report,
            validation,
        )
    end
end
