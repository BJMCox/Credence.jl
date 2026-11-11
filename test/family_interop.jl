@testset "Positive, simplex, and heavy-tail prior coordinates" begin
    problem = Elicitation(
        (),
        (
            :rate => (:gamma => GammaFamily(shape=2.0, scale=3.0),),
            :weights => (
                :dirichlet =>
                    DirichletFamily(3; mean=[0.2, 0.3, 0.5], concentration=10.0),
            ),
            :beta => (
                :student => MvStudentTFamily(
                    2;
                    df=4.0,
                    starts=(location=[0.5, -0.2], log_scales=log.([1.2, 0.8]), lower=[0.3]),
                ),
            ),
        );
        combinations=(
            :joint => Independent(:rate=>:gamma, :weights=>:dirichlet, :beta=>:student),
        ),
    )
    prior = only(fit(Xoshiro(901), problem).results).measure
    from = Plaice.from_unconstrained_vec(prior)
    z = [log(4.0), -0.3, 0.4, 1.0, -0.5]
    point, correction = with_logabsdet_jacobian(from, z)
    @test point.rate ≈ 4.0
    @test sum(point.weights) ≈ 1.0 && all(>(0), point.weights)
    @test point.beta ≈ z[4:5]
    @test Plaice.to_unconstrained_vec(prior)(point) ≈ z
    @test MeasureBase.logdensityof(prior, point) ≈
          logpdf(Gamma(2.0, 3.0), point.rate) +
          logpdf(Dirichlet([2.0, 3.0, 5.0]), point.weights) +
          logpdf(
              Distributions.MvTDist(4.0, [0.5, -0.2], [1.44 0.36; 0.36 0.73]),
              point.beta,
          )
    coordinates(v) = begin
        value = from(v)
        vcat(value.rate, value.weights[1:2], value.beta)
    end
    @test correction ≈ first(logabsdet(ForwardDiff.jacobian(coordinates, z)))
    density(v) = begin
        value, jacobian = with_logabsdet_jacobian(from, v)
        MeasureBase.logdensityof(prior, value) + jacobian
    end
    step = cbrt(eps(Float64))
    finite_gradient = map(eachindex(z)) do i
        delta = zeros(length(z))
        delta[i] = step
        (density(z + delta) - density(z - delta)) / (2step)
    end
    @test ForwardDiff.gradient(density, z) ≈ finite_gradient rtol=sqrt(step) atol=sqrt(step)
end
