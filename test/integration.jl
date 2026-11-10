using Plaice
import ChangesOfVariables: with_logabsdet_jacobian

@testset "Native joint-factor composition and support charts" begin
    covariance = [1.0 0.4; 0.4 0.8]
    joint = Joint(:a, :b)
    catalog = (
        joint=>(
            Candidate(:normal, _ -> asmeasure(MvNormal([0.2, -0.1], covariance)), [0.0]),
        ),
        :sigma=>(Candidate(:lognormal, _ -> asmeasure(LogNormal(-0.3, 0.4)), [0.0]),),
    )
    problem = Elicitation(
        (joint=>(Approximate(:covariance, CovarianceMatrix(), covariance; scale=1.0),),),
        catalog;
        combinations=(:joint=>Independent(joint=>:normal, :sigma=>:lognormal),),
    )
    report = fit(Xoshiro(1), problem)
    @test only(report.results).error === nothing
    checks = validate(Xoshiro(2), report)
    @test only(only(checks.checks)).estimate.value == covariance
    selection = select(
        report,
        (:joint, 1);
        validation=checks,
        rationale="The joint covariance is preserved",
    )
    law = measure(selection)
    z = [0.4, -0.2, log(0.8)]
    point = (a=z[1], b=z[2], sigma=exp(z[3]))
    @test MeasureBase.logdensityof(law, point) ≈
          logpdf(MvNormal([0.2, -0.1], covariance), z[1:2])+logpdf(
        LogNormal(-0.3, 0.4),
        point.sigma,
    )
    from = Plaice.from_unconstrained_vec(law)
    @test Plaice.to_unconstrained_vec(law)(from(z)) ≈ z
    restored, jacobian = with_logabsdet_jacobian(from, z)
    @test restored == point
    @test jacobian ≈ z[3]
    density(v) = begin
        value, correction = with_logabsdet_jacobian(from, v)
        MeasureBase.logdensityof(law, value)+correction
    end
    expected = vcat(-(covariance \ (z[1:2]-[0.2, -0.1])), -(z[3]+0.3)/0.4^2)
    @test ForwardDiff.gradient(density, z) ≈ expected
    @test [optic(point) for optic in Plaice.optic_vec(law)] ≈ [point.a, point.b, point.sigma]
    draws = rand(Xoshiro(3), law^4)
    @test length(draws) == 4
    @test MeasureBase.logdensities(law, draws) ≈
          [MeasureBase.logdensityof(law, x) for x in draws]
    @test MeasureBase.logdensityof(law^4, draws) ≈
          sum(MeasureBase.logdensityof(law, x) for x in draws)
    @test_throws DimensionMismatch MeasureBase.logdensityof(law^4, draws[1:2])
    @test_throws DimensionMismatch MeasureBase.logdensityof(law^(2, 2), draws)
end

@testset "IID arrays and indexed factor composition" begin
    problem = Elicitation(
        (:A=>(Approximate(:mean, Mean(), ones(2, 2); scale=1.0),),),
        (:A=>(:iid=>IIDFamily(LogNormalFamily(location=0.0), (2, 2)),),),
    )
    report = fit(Xoshiro(1), problem)
    law = first(report.results).measure
    chart = Plaice.from_unconstrained_vec(law)
    z = [0.2, -0.3, 0.4, 0.7]
    point, logjacobian = with_logabsdet_jacobian(chart, z)
    @test point ≈ reshape(exp.(z), 2, 2)
    @test logjacobian ≈ sum(z)
    @test Plaice.to_unconstrained_vec(law)(point) ≈ z
    @test [optic(point) for optic in Plaice.optic_vec(law)] ≈ vec(point)
    candidates = (
        @vn(B[1])=>(Candidate(:normal, _ -> asmeasure(Normal(1.0, 1.0)), [0.0]),),
        @vn(B[2])=>(Candidate(:normal, _ -> asmeasure(Normal(2.0, 1.0)), [0.0]),),
    )
    combined = Elicitation(
        (:B=>(Approximate(:mean, Mean(), [1.0, 2.0]; scale=1.0),),),
        candidates;
        combinations=(:both=>Independent(@vn(B[1])=>:normal, @vn(B[2])=>:normal),),
    )
    result = only(fit(Xoshiro(2), combined).results)
    @test result.error === nothing
    @test mean(marginals(result.measure).B) == [1.0, 2.0]
    @test Plaice.from_vec(result.measure)([1.0, 2.0]) == (B=[1.0, 2.0],)
    block = Joint(@vn(B[1]), @vn(B[2]))
    covariance = [1.0 0.4; 0.4 1.0]
    named = Elicitation(
        (),
        (
            block => (
                Candidate(
                    :block,
                    _ -> productmeasure((B=asmeasure(MvNormal(zeros(2), covariance)),)),
                    [0.0],
                ),
            ),
            @vn(B[3]) => (Candidate(:tail, _ -> asmeasure(Normal()), [0.0]),),
        );
        combinations=(:all => Independent(block=>:block, @vn(B[3])=>:tail),),
    )
    law = only(fit(Xoshiro(3), named).results).measure
    point = (B=[0.2, -0.1, 0.4],)
    @test MeasureBase.logdensityof(law, point) ≈
          logpdf(MvNormal(zeros(2), covariance), point.B[1:2])+logpdf(Normal(), point.B[3])
    @test Plaice.from_vec(law)(Plaice.to_vec(law)(point)) == point
end
