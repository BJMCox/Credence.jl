using Distributions
using MeasureBase

@testset "Discrete beliefs preserve atoms" begin
    bernoulli = MeasureBase.asmeasure(Bernoulli(0.5))
    @test evaluate(Quantile(0.5), bernoulli).value == 0
    @test evaluate(Quantile(nextfloat(0.5)), bernoulli).value == 1
    @test evaluate(Probability(-Inf, 0), bernoulli).value == 0.5
    @test evaluate(Probability(0, 1), bernoulli).value == 0.5

    counts = MeasureBase.asmeasure(Poisson(2.0))
    @test evaluate(Quantile(0.5), counts).value == 2
    @test evaluate(Probability(-1, 0), counts).value ≈ exp(-2)
    @test evaluate(Probability(0, 2), counts).value ≈ 4exp(-2)
    @test evaluate(Probability(20, 21), counts).value ≈
          exp(-2) * 2.0^21 / Float64(factorial(big(21)))

    separated = MeasureBase.asmeasure(DiscreteNonParametric([0, 2], [0.5, 0.5]))
    @test evaluate(Quantile(0.5), separated).value == 0
    @test evaluate(Quantile(nextfloat(0.5)), separated).value == 2
    @test evaluate(Probability(0, 1), separated).value == 0
end
