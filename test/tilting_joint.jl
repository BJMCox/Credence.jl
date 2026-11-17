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

@testset "Joint tilting with fixed marginals" begin
    reference = productmeasure((gain=StdUniform(), drift=StdUniform(), offset=StdUniform()))
    both = Probability(offset=(1/8, 7/8), gain=(1/4, 1/2))
    beliefs = (Hard(:both, both, EqualTo(7/32)),)
    method = BoundedKL(lower=0.5, upper=2)
    report = tilt(reference, beliefs, method; preserve_marginals=true)
    validation = validate(Xoshiro(113), report)
    choice = select(report; validation, rationale="Keep the calibration marginals")
    prior = measure(choice)
    @test evaluate(both, prior).value == 7/32
    @test marginals(prior) == marginals(reference)
    overlap = Probability(gain=(1/8, 3/8), offset=(3/4, 1))
    @test evaluate(overlap, prior).value ≈ 17/288
    mixed = Probability(gain=(1/8, 3/8), drift=(1/2, 7/8), offset=(3/4, 1))
    @test evaluate(mixed, prior).value ≈ 17/768
    partial = Probability(gain=(1/8, 3/8), drift=(1/2, 7/8))
    @test evaluate(partial, prior).value == 3/32
    for (gain, offset, ratio) in
        ((1/8, 1/16, 7/6), (3/8, 1/16, 1/2), (1/8, 1/2, 17/18), (3/8, 1/2, 7/6))
        point = (; gain, drift=0.4, offset)
        @test logdensityof(prior, point) ≈ logdensityof(reference, point) + log(ratio)
    end
    noop = (Hard(:both, both, Between(0.15, 0.20)),)
    @test measure(tilt(reference, noop, method; preserve_marginals=true)) === reference
    impossible = (Hard(:both, both, EqualTo(1/4)),)
    @test_throws ArgumentError tilt(reference, impossible, method; preserve_marginals=true)

    mktempdir() do directory
        path = joinpath(directory, "fixed-marginals.toml")
        save_session(path, choice)
        restored = only(load_session(path).selections)
        @test evaluate(overlap, measure(restored)).value == evaluate(overlap, prior).value
        rng = Xoshiro(114)
        @test rand(Xoshiro(114), measure(restored)^32) == [rand(rng, prior) for _ in 1:32]
        @test flatppl(restored) == flatppl(choice)
    end

    halves = Probability(gain=(1/2, 1), offset=(1/2, 1))
    for (target, lower) in ((0.275, 0.9), (0.001, 0.0))
        fit = tilt(
            reference,
            (Hard(:both, halves, EqualTo(target)),),
            BoundedKL(; lower, upper=2);
            preserve_marginals=true,
        )
        @test only(only(validate(Xoshiro(115), fit).checks)).status == :satisfied
    end
    anticorrelated = measure(
        tilt(
            reference,
            (Hard(:both, halves, EqualTo(0)),),
            BoundedKL(lower=0, upper=2);
            preserve_marginals=true,
        ),
    )
    rng = Xoshiro(116)
    @test all(
        x -> (x.gain > 1/2) != (x.offset > 1/2),
        [rand(rng, anticorrelated) for _ in 1:32],
    )
end
