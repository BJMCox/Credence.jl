using Optimization, OptimizationOptimJL

mutable struct TiltTestRNG <: AbstractRNG
    values::Vector{Float64}
    index::Int
end
function Random.rand(rng::TiltTestRNG, ::Type{Float64})
    rng.index += 1
    rng.values[mod1(rng.index, length(rng.values))]
end
function Random.rand!(rng::TiltTestRNG, values::AbstractArray{Float64})
    for i in eachindex(values)
        values[i] = rand(rng, Float64)
    end
    values
end

@testset "Bounded KL" begin
    method = BoundedKL(lower=1/2, upper=2)
    @testset "Analytic event law and sampling" begin
        belief = Hard(:positive, Probability(0, Inf), EqualTo(0.7))
        report = tilt(StdNormal(), (belief,), method)
        prior = measure(report)
        @test evaluate(Probability(0, Inf), prior).value ≈ 0.7
        @test logdensityof(prior, 1.0) ≈ log(1.4) - (log(2pi)+1)/2
        @test logdensityof(prior, -1.0) ≈ log(0.6) - (log(2pi)+1)/2
        @test evaluate(Quantile(0.3), prior).value ≈ 0 atol=1e-14
        @test report.divergence.value ≈ 0.3log(0.6) + 0.7log(1.4)
        draws = rand(Xoshiro(510), prior^8192)
        @test abs(count(>(0), draws)/length(draws)-0.7) < 6sqrt(0.7*0.3/length(draws))
        checks = validate(Xoshiro(11), report)
        @test only(only(checks.checks)).status == :satisfied
        @test measure(
            select(report; validation=checks, rationale="Analytic event constraint"),
        ) === prior

        fixed = MeasureBase.Dirac(0.0)
        unchanged = tilt(fixed, (Hard(:inside, Probability(-1, 1), EqualTo(1)),), method)
        @test measure(unchanged) === fixed
        @test only(only(validate(Xoshiro(12), unchanged).checks)).status == :satisfied
        for reference in (StdNormal(), asmeasure(Cauchy()))
            tilted = tilt(reference, (belief,), method)
            problem = Elicitation(
                (Approximate(:mean, Mean(), 0.0; scale=1.0),),
                (Candidate(:tilted, _ -> measure(tilted), [0.0]),),
            )
            checks =
                validate(Xoshiro(15), fit(Xoshiro(16), problem); sampling=MonteCarlo(1024))
            expected = reference isa MeasureBase.StdNormal ? :target : :undefined
            @test only(only(checks.checks)).status == expected
        end
        @test_throws ArgumentError tilt(2*StdNormal(), (), method)
        @test_throws ArgumentError tilt(
            StdUniform(),
            (Hard(:null, Probability(2, 3), EqualTo(0.1)),),
            method,
        )
        @test_throws ArgumentError tilt(
            StdUniform(),
            (Hard(:upper, Probability(0, 0.9), EqualTo(0.99)),),
            method,
        )
    end

    @testset "Zero mass and boundary solutions" begin
        boundary = tilt(
            StdUniform(),
            (Hard(:mean, Mean(), EqualTo(5/8)),),
            BoundedKL(lower=0.5, upper=1.5),
        )
        @test mean(measure(boundary)) == 5/8
        @test exp(logdensityof(measure(boundary), 0.25)) ≈ 0.5
        @test exp(logdensityof(measure(boundary), 0.75)) ≈ 1.5
        @test_throws ArgumentError tilt(
            StdUniform(),
            (Hard(:mean, Mean(), AtLeast(0.7)),),
            BoundedKL(lower=0.5, upper=1.5),
        )
        cut = tilt(
            StdUniform(),
            (Hard(:upper, Probability(0.5, Inf), EqualTo(1)),),
            BoundedKL(lower=0, upper=2),
        )
        @test rand(TiltTestRNG([0.25, 0.0, 0.75, 0.5], 0), measure(cut)) == 0.75
    end

    @testset "Analytic continuous moment oracle" begin
        target = 2-1/log(2)
        report = tilt(
            StdUniform(),
            (Hard(:mean, Mean(), EqualTo(target)),),
            method,
            BFGS();
            g_tol=1e-11,
        )
        prior = measure(report)
        @test mean(prior) ≈ target atol=1e-10
        @test logdensityof(prior, 0.4) ≈ log(log(2))+0.4log(2) atol=1e-10
        @test report.divergence.value ≈ log(log(2))+2log(2)-1 atol=1e-10
        @test report.normalization.evidence == :analytic
        for p in (0.1, 0.5, 0.9)
            @test evaluate(Quantile(p), prior).value ≈ log1p(p)/log(2) atol=1e-10
        end
        @test invsmf(prior, 0.0) == 0.0
        @test invsmf(prior, 1.0) ≈ 1.0
        @test method.lower <=
              first(report.ratio_bounds) <=
              last(report.ratio_bounds) <=
              method.upper
        interval = tilt(
            StdUniform(),
            (Hard(:mean, Mean(), Between(0.55, 0.6)),),
            method,
            Fminbox(BFGS());
            g_tol=1e-10,
        )
        @test 0.55 <= mean(measure(interval)) <= 0.6
        @test mean(measure(interval)) ≈ 0.55 atol=1e-8
        identity_view =
            tilt(StdUniform(), (Hard(:mean, Expectation(identity), EqualTo(0.5)),), method)
        @test measure(identity_view) isa MeasureBase.StdUniform
        @test_throws ArgumentError tilt(
            asmeasure(Cauchy()),
            (Hard(:mean, Expectation(identity), EqualTo(0.2)),),
            method,
            BFGS(),
        )
    end

    @testset "Numerical normalization stays explicit" begin
        beliefs = (Hard(:mean, Mean(), EqualTo(0.2)),)
        report = tilt(StdNormal(), beliefs, method, BFGS(); g_tol=1e-10)
        prior = measure(report)
        @test report.normalization.evidence == :estimated
        @test_throws ArgumentError logdensityof(prior, 0.4)
        estimate = logdensity_estimate(report, 0.4)
        translated = tilt(
            asmeasure(Normal(125.0, 1.0)),
            (Hard(:mean, Mean(), EqualTo(125.2)),),
            method,
            BFGS();
            g_tol=1e-10,
        )
        @test logdensity_estimate(translated, 125.4).value ≈ estimate.value atol=1e-7
        @test estimate.evidence == :estimated
        @test estimate.lower <= estimate.value <= estimate.upper
        @test first(report.ratio_bounds) ≈ method.lower/report.normalization.upper
        @test last(report.ratio_bounds) ≈ method.upper/report.normalization.lower
        draws = rand(Xoshiro(610), prior^8192)
        @test abs(mean(draws)-0.2) < 6std(draws)/sqrt(length(draws))
        checks = validate(Xoshiro(14), report; sampling=MonteCarlo(4096))
        @test only(only(checks.checks)).status == :inconclusive
        @test_throws ArgumentError select(
            report;
            validation=checks,
            rationale="Unresolved normalization",
        )
    end

    @testset "Mixed expectations and dual gradient" begin
        beliefs = (
            Hard(:mean, Mean(), EqualTo(0.55)),
            Hard(:upper, Probability(0.5, 1), EqualTo(0.6)),
        )
        # Constant weights 0.8 and 1.2 give both targets and minimize KL within each cell.
        report = tilt(StdUniform(), beliefs, method, BFGS(); g_tol=1e-10)
        @test logdensity_estimate(report, 0.25).value ≈ log(0.8) atol=1e-8
        @test logdensity_estimate(report, 0.75).value ≈ log(1.2) atol=1e-8
        @test report.divergence.value ≈ 0.4log(0.8)+0.6log(1.2) atol=1e-8
        @test report.optimality == :numerical

        data = Credence._tilt_data(
            StdUniform(),
            beliefs,
            method,
            Quadrature(rtol=1e-11, atol=1e-13),
        )
        point = [-0.8, 1.4, 0.3]
        _, gradient, _ = Credence._tilt_dual(data, point)
        h = 1e-5
        finite_difference = map(eachindex(point)) do i
            delta = zeros(length(point))
            delta[i] = h
            (
                first(Credence._tilt_dual(data, point+delta)) -
                first(Credence._tilt_dual(data, point-delta))
            ) / (2h)
        end
        @test gradient ≈ finite_difference atol=1e-7
        @test_throws ArgumentError tilt(StdUniform(), beliefs, method, BFGS(); atol=0)
    end
end
