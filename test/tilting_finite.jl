using Credence, MeasureBase, Optimization, OptimizationOptimJL, Test, Random

@testset "Finite bounded KL" begin
    reference = FiniteReference((0.0, 1.0, 2.0), (1, 2, 1))
    method = BoundedKL(lower=1/2, upper=2)
    masses(report, points=reference.points) =
        [exp(logdensityof(measure(report), x)) for x in points]

    @testset "Reference atoms and relative weights" begin
        @test_throws ArgumentError FiniteReference((0.0, -0.0), (1, 1))
        @test_throws ArgumentError FiniteReference((NaN,), (1,))
        @test_throws ArgumentError FiniteReference((0, 1), (0, 0))
        @test_throws ArgumentError FiniteReference((0, 1), (1, -1))
        @test FiniteReference(0:2, [5e307, 1e308, 5e307]).weights == reference.weights
    end

    @testset "Active cap and equivalent constraints" begin
        target = Hard(:mean, Mean(), EqualTo(27/20))
        report = tilt(reference, (target,), method, IPNewton(); abstol=1e-12)
        expected = [1/8, 2/5, 19/40]
        @test masses(report) ≈ expected atol=1e-8
        @test report.optimality == :numerical
        @test report.feasibility in (:feasible, :within_tolerance)
        @test report.normalization.evidence == :analytic
        @test report.divergence.value ≈ sum(expected .* log.([1/2, 4/5, 19/10]))
        @test report.fit.results[1].solver.dual_gap <= 1e-8

        scaled_reference = FiniteReference(reference.points, (5e307, 1e308, 5e307))
        scaled = tilt(scaled_reference, (target,), method, IPNewton(); abstol=1e-12)
        @test masses(scaled) ≈ expected atol=1e-8

        equivalent = (
            target,
            Hard(:scaled, Expectation(x -> 1000x+7), EqualTo(1357)),
            Hard(:negative, Expectation(x -> -x), EqualTo(-27/20)),
            Hard(:inactive, Mean(), AtLeast(1.1)),
        )
        duplicated = tilt(reference, equivalent, method, IPNewton(); abstol=1e-12)
        @test masses(duplicated) ≈ expected atol=1e-8
        @test duplicated.divergence.value ≈ report.divergence.value atol=1e-10

        active = tilt(
            reference,
            (Hard(:minimum, Mean(), AtLeast(27/20)),),
            method,
            IPNewton();
            abstol=1e-12,
        )
        @test masses(active) ≈ expected atol=1e-8
        unchanged =
            tilt(reference, (Hard(:loose, Mean(), Between(0.8, 1.2)),), method, IPNewton())
        @test masses(unchanged) ≈ reference.weights
        @test unchanged.divergence.value ≈ 0 atol=1e-15
    end

    @testset "Boundary and feasibility witnesses" begin
        maximum_mean =
            tilt(reference, (Hard(:max, Mean(), EqualTo(11/8)),), method, IPNewton())
        @test masses(maximum_mean) ≈ [1/8, 3/8, 1/2]
        @test maximum_mean.optimality == :analytic
        @test_throws ArgumentError tilt(
            reference,
            (Hard(:impossible, Mean(), AtLeast(1.4)),),
            method,
            IPNewton(),
        )
        @test_throws ArgumentError tilt(
            reference,
            (Hard(:one, Mean(), EqualTo(1)), Hard(:two, Expectation(x -> 2x), EqualTo(3))),
            method,
            IPNewton(),
        )

        event_reference = FiniteReference((0, 1, 100), (3, 1, 0))
        event = (Hard(:event, Probability(==(1)), EqualTo(1)),)
        conditioned = tilt(event_reference, event, BoundedKL(lower=0, upper=4), IPNewton())
        @test masses(conditioned, (0, 1, 100)) ≈ [0, 1, 0]
        @test conditioned.divergence.value ≈ log(4)
        @test conditioned.ratio_bounds == (0.0, 4.0)
        rng = Xoshiro(917)
        @test all(==(1), [rand(rng, measure(conditioned)) for _ in 1:10])
        @test_throws ArgumentError tilt(
            event_reference,
            event,
            BoundedKL(lower=0.1, upper=4),
            IPNewton(),
        )

        tied = FiniteReference((0, 1, 2), (1, 1, 2))
        tied_result = tilt(
            tied,
            (Hard(:event, Probability(x -> x > 0), EqualTo(1)),),
            BoundedKL(lower=0, upper=2),
            IPNewton(),
        )
        @test masses(tied_result, tied.points) ≈ [0, 1/3, 2/3]
    end

    @testset "Several constraints and redundant equalities" begin
        reference = FiniteReference(0:3, ones(4))
        beliefs = (
            Hard(:first, Probability(==(0)), EqualTo(0.1)),
            Hard(:second, Probability(==(1)), EqualTo(0.2)),
            Hard(:sum, Expectation(x -> Int(x == 0)+Int(x == 1)), EqualTo(0.3)),
        )
        result = tilt(
            reference,
            beliefs,
            BoundedKL(lower=0.25, upper=2),
            IPNewton();
            abstol=1e-12,
        )
        @test masses(result, reference.points) ≈ [0.1, 0.2, 0.35, 0.35] atol=1e-8
        @test result.optimality == :numerical
        @test result.fit.results[1].solver.residual <= 1e-8

        capped = tilt(
            reference,
            (
                Hard(:first, Probability(==(0)), EqualTo(0.07)),
                Hard(:mean, Mean(), EqualTo(2.25)),
            ),
            BoundedKL(lower=0.25, upper=2),
            IPNewton();
            abstol=1e-12,
        )
        @test masses(capped, reference.points) ≈ [0.07, 0.11, 0.32, 0.5] atol=1e-8
        @test capped.optimality == :numerical

        inequalities = tilt(
            reference,
            (
                Hard(:first, Probability(==(0)), AtLeast(0.4)),
                Hard(:second, Probability(==(1)), AtLeast(0.3)),
            ),
            BoundedKL(lower=0.25, upper=2),
            IPNewton();
            abstol=1e-12,
        )
        @test masses(inequalities, reference.points) ≈ [0.4, 0.3, 0.15, 0.15] atol=1e-8

        limited = tilt(
            FiniteReference((0, 1, 2), (1, 2, 1)),
            (Hard(:mean, Mean(), EqualTo(27/20)),),
            method,
            IPNewton();
            maxiters=0,
        )
        @test limited.feasibility != :infeasible
        @test limited.optimality == :unresolved
    end

    @testset "Several constraints on a boundary face" begin
        reference = FiniteReference(0:3, ones(4))
        capped = tilt(
            reference,
            (
                Hard(:first, Probability(==(0)), EqualTo(0.125)),
                Hard(:second, Probability(==(1)), EqualTo(0.125)),
            ),
            BoundedKL(lower=0.5, upper=1.5),
            IPNewton(),
        )
        @test masses(capped, reference.points) ≈ [0.125, 0.125, 0.375, 0.375]
        @test capped.optimality == :analytic

        conditioned = tilt(
            reference,
            (
                Hard(:absent, Probability(==(0)), AtMost(0.0)),
                Hard(:second, Probability(==(1)), EqualTo(0.2)),
            ),
            BoundedKL(lower=0, upper=3),
            IPNewton();
            abstol=1e-12,
        )
        @test masses(conditioned, reference.points) ≈ [0, 0.2, 0.4, 0.4] atol=1e-8
        @test masses(conditioned, (0,)) == [0]
        @test conditioned.optimality == :numerical
        @test conditioned.fit.results[1].solver.dual_gap <= 1e-8
        @test conditioned.divergence.value ≈ 0.2log(0.8)+0.8log(1.6) atol=1e-8

        determined = tilt(
            reference,
            Tuple(Hard(i, Probability(==(i)), EqualTo((i+1)/10)) for i in 0:2),
            BoundedKL(lower=0, upper=3),
            IPNewton(),
        )
        @test masses(determined, reference.points) ≈ [0.1, 0.2, 0.3, 0.4]
        @test determined.optimality == :analytic

        combined = tilt(
            FiniteReference(0:4, ones(5)),
            (
                Hard(:a, Probability(x -> x in (0, 1)), EqualTo(0.3)),
                Hard(:b, Probability(x -> x in (1, 2)), EqualTo(0.3)),
                Hard(:c, Probability(x -> x in (0, 2)), EqualTo(0.6)),
            ),
            BoundedKL(lower=0, upper=4),
            IPNewton();
            abstol=1e-12,
            maxiters=100,
        )
        @test combined.optimality == :unresolved
    end
end
