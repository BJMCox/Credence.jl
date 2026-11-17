@testset "Joint probability tilting" begin
    reference = productmeasure((gain=StdNormal(), offset=StdNormal()))
    both = Probability(gain=(0, Inf), offset=(0, Inf))
    @test_throws ArgumentError Probability(gain=0)
    method = BoundedKL(lower=0.5, upper=2)
    @test evaluate(both, reference).value == 0.25
    report = tilt(reference, (Hard(:both, both, Between(0.4, 0.45)),), method)
    validation = validate(Xoshiro(111), report)
    selection = select(report; validation, rationale="Reviewed joint calibration")
    prior = measure(selection)
    @test evaluate(both, prior).value ≈ 0.4
    @test evaluate(Probability(gain=(0, Inf)), prior).value ≈ 0.6
    @test evaluate(Probability(offset=(0, Inf)), prior).value ≈ 0.6
    @test evaluate(Probability(gain=(-Inf, 0), offset=(0, Inf)), prior).value ≈ 0.2
    @test report.ratio_bounds[1] ≈ 0.8
    @test report.ratio_bounds[2] ≈ 1.6
    @test report.divergence.value ≈ 0.4log(1.6) + 0.6log(0.8)
    for (point, ratio) in (((gain=0.3, offset=0.4), 1.6), ((gain=0.0, offset=0.4), 0.8))
        @test logdensityof(prior, point) ≈ logdensityof(reference, point) + log(ratio)
    end

    rng = Xoshiro(117)
    @test rand(Xoshiro(117), prior^32) == [rand(rng, prior) for _ in 1:32]

    conditioned =
        tilt(reference, (Hard(:both, both, EqualTo(1)),), BoundedKL(lower=0, upper=4))
    @test evaluate(both, measure(conditioned)).value == 1
    rng = Xoshiro(112)
    @test all(
        x -> x.gain > 0 && x.offset > 0,
        [rand(rng, measure(conditioned)) for _ in 1:64],
    )

    mktempdir() do directory
        path = joinpath(directory, "calibration.toml")
        save_session(path, selection)
        restored = only(load_session(path).selections)
        @test restored.report.ratio_bounds == report.ratio_bounds
        @test evaluate(both, measure(restored)).value == evaluate(both, prior).value
        @test flatppl(restored) == flatppl(selection)
    end
end
