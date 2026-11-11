using Plaice, ForwardDiff, LinearAlgebra
using HeterogeneousComputing: GenContext

@testset "Sparse compound allocation coordinates" begin
    family = DirichletLaplaceFamily(32; scale=1.0)
    prior =
        only(fit(Xoshiro(3100), Elicitation((), (:theta => (:dl => family,),))).results).measure
    rng = Xoshiro(3132)
    point = rand(rng, prior)
    for _ in 2:9
        point = rand(rng, prior)
    end
    to_coordinates = Plaice.to_unconstrained_vec(prior)
    from_coordinates = Plaice.from_unconstrained_vec(prior)
    coordinates = to_coordinates(point)
    restored = from_coordinates(coordinates)
    @test all(isfinite, coordinates)
    @test all(isapprox.(restored.allocation, point.allocation; rtol=1e-12, atol=0))
    @test logdensityof(prior, restored) ≈ logdensityof(prior, point)

    family = DirichletLaplaceFamily(3; scale=0.7)
    prior =
        only(fit(Xoshiro(3101), Elicitation((), (:theta => (:dl => family,),))).results).measure
    chart = Plaice.from_unconstrained_vec(prior)
    coordinates = [0.2, -0.3, 0.4, log(1.3), -0.5, 0.7]
    reduced =
        coordinates -> begin
            value = chart(coordinates)
            vcat(value.coefficients, value.global_scale, value.allocation[1:2])
        end
    jacobian = ForwardDiff.jacobian(reduced, coordinates)
    @test last(Plaice.with_logabsdet_jacobian(chart, coordinates)) ≈
          first(logabsdet(jacobian))

    bad_point = (coefficients=zeros(3), global_scale=1.0, allocation=[0.0, 0.4, 0.6])
    @test_throws ArgumentError Plaice.to_unconstrained_vec(prior)(bad_point)

    family = R2D2Family(1; noise_scale=1.0, predictor_scales=[1.0])
    prior =
        only(fit(Xoshiro(3103), Elicitation((), (:theta => (:r2d2 => family,),))).results).measure
    coordinates = [0.2, -0.5]
    point = Plaice.from_unconstrained_vec(prior)(coordinates)
    @test point.allocation == [1.0]
    @test Plaice.to_unconstrained_vec(prior)(point) ≈ coordinates

    setprecision(BigFloat, 256) do
        n = 128
        family = DirichletLaplaceFamily(n; concentration=BigFloat(1)/n, scale=BigFloat(1))
        prior =
            only(fit(Xoshiro(3102), Elicitation((), (:theta => (:dl => family,),))).results).measure
        point = rand(GenContext{BigFloat}(Xoshiro(3428)), prior)
        coordinates = Plaice.to_unconstrained_vec(prior)(point)
        restored = Plaice.from_unconstrained_vec(prior)(coordinates)
        @test isfinite(logdensityof(prior, point)) && all(isfinite, coordinates)
        @test all(isapprox.(restored.allocation, point.allocation; rtol=big"1e-70", atol=0))
    end
end
