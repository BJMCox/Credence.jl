using Plaice
using ForwardDiff

@testset "Tilted priors retain support charts" begin
    report = tilt(
        StdNormal(),
        (Hard(:positive, Probability(0, Inf), Between(0.68, 0.72)),),
        BoundedKL(lower=0.5, upper=2),
    )
    validation = validate(Xoshiro(702), report)
    prior = measure(select(report; validation, rationale="Support chart integration"))
    restored, jacobian =
        Plaice.with_logabsdet_jacobian(Plaice.from_unconstrained_vec(prior), [0.4])
    @test restored == 0.4
    @test jacobian == 0
    @test Plaice.to_vec(prior)(restored) == [0.4]
    @test only(Plaice.optic_vec(prior))(restored) == restored
    @test ForwardDiff.derivative(x -> logdensityof(prior, x), 0.4) ≈ -0.4

    conditioned = measure(
        tilt(
            StdNormal(),
            (Hard(:positive, Probability(0, Inf), EqualTo(1)),),
            BoundedKL(lower=0, upper=2),
        ),
    )
    point, jacobian =
        Plaice.with_logabsdet_jacobian(Plaice.from_unconstrained_vec(conditioned), [0.4])
    @test point ≈ exp(0.4)
    @test jacobian ≈ 0.4
    @test Plaice.to_unconstrained_vec(conditioned)(point) ≈ [0.4]

    tails = measure(
        tilt(
            StdUniform(),
            (Hard(:middle, Probability(0.25, 0.75), EqualTo(0)),),
            BoundedKL(lower=0, upper=2),
        ),
    )
    @test_throws ArgumentError Plaice.to_unconstrained_vec(tails)

    left_tail = measure(
        tilt(
            StdNormal(),
            (Hard(:positive, Probability(0, Inf), EqualTo(0)),),
            BoundedKL(lower=0, upper=2),
        ),
    )
    @test Plaice.from_unconstrained_vec(left_tail)([0.4]) ≈ -exp(0.4)
end
