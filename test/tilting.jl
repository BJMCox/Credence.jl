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
        cut = tilt(
            StdUniform(),
            (Hard(:upper, Probability(0.5, Inf), EqualTo(1)),),
            BoundedKL(lower=0, upper=2),
        )
        @test rand(TiltTestRNG([0.25, 0.0, 0.75, 0.5], 0), measure(cut)) == 0.75
    end

end
