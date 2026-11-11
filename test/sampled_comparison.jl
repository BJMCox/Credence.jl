@testset "Sampled validation and comparison" begin
    beliefs = (Approximate(:event, Probability(x -> x > 0), 0.0; scale=1.0),)
    problem = Elicitation(
        beliefs,
        (
            Candidate(:near, _ -> asmeasure(Normal(-3.0, 1.0)), [0.0]),
            Candidate(:far, _ -> asmeasure(Normal(3.0, 1.0)), [0.0]),
        ),
    )
    report = fit(Xoshiro(1), problem; sampling=MonteCarlo(512))
    first_validation = validate(Xoshiro(2), report)
    second_validation = validate(Xoshiro(3), report)
    @test first_validation.seed != second_validation.seed
    @test first_validation.checks[1][1].status == :target
    @test validate(Xoshiro(2), report; sampling=nothing).checks[1][1].status == :undefined
    @test_throws ArgumentError compare(report)
    @test compare(report; validation=first_validation).dominates ==
          [((:near, 1), (:far, 1))]
    @test compare(Xoshiro(2), report).dominates == [((:near, 1), (:far, 1))]

    overlapping = fit(
        Xoshiro(4),
        Elicitation(beliefs, (Candidate(:same, _ -> StdNormal(), ([0.0], [1.0])),));
        sampling=MonteCarlo(512),
    )
    @test compare(Xoshiro(5), overlapping).unresolved == [((:same, 1), (:same, 2))]
end

@testset "Simultaneous validation budget" begin
    law = productmeasure([asmeasure(Normal()), asmeasure(Normal())])
    problem = Elicitation(
        (
            Hard(:event, Probability(x -> x[1] > 0), Between(0.0, 1.0)),
            Approximate(:other, Probability(x -> x[2] > 0), 0.5; scale=1.0),
            Approximate(:quantiles, Quantile(0.5), [0.0, 0.0]; scale=1.0),
        ),
        (Candidate(:joint, _ -> law, ([0.0], [1.0])),),
    )
    n, alpha = 128, 0.12
    validation = validate(Xoshiro(7), fit(Xoshiro(6), problem); plan=MonteCarlo(n; alpha))
    checks = first(validation.checks)
    stream = Xoshiro(validation.seed)
    draws = [rand(stream, law) for _ in 1:n]
    radius = sqrt(log(2 / (alpha / 6)) / (2n))
    for (i, predicate) in enumerate((x -> x[1] > 0, x -> x[2] > 0))
        probability = count(predicate, draws) / n
        @test checks[i].estimate.lower ≈ max(0, probability - radius)
        @test checks[i].estimate.upper ≈ min(1, probability + radius)
    end
    quantile_radius = sqrt(log(2 * 2 / (alpha / 6)) / (2n))
    ordered = sort(reduce(hcat, draws); dims=2)
    @test checks[3].estimate.lower == ordered[:, ceil(Int, n * (0.5 - quantile_radius))]
    @test checks[3].estimate.upper == ordered[:, ceil(Int, n * (0.5 + quantile_radius))]
end

@testset "Discrepancy enclosures" begin
    candidates = (
        Candidate(:left, _ -> asmeasure(Normal(1.0, 1.0)), [0.0]),
        Candidate(:right, _ -> asmeasure(Normal(3.0, 1.0)), [0.0]),
    )
    array_statistic = Statistic(
        law -> Estimate(
            fill(mean(law), 3);
            lower=fill(mean(law)-0.1, 3),
            upper=fill(mean(law)+0.1, 3),
            evidence=:certified,
        ),
    )
    problem = Elicitation(
        (
            Approximate(
                :vector,
                array_statistic,
                [0.0, missing, 4.0];
                scale=[1.0, 1.0, 2.0],
            ),
        ),
        candidates,
    )
    @test compare(fit(Xoshiro(8), problem)).dominates == [((:left, 1), (:right, 1))]

    for evidence in (:certified, :estimated)
        statistic = Statistic(
            law ->
                Estimate(mean(law); lower=mean(law)-0.1, upper=mean(law)+0.1, evidence),
        )
        tradeoff = Elicitation(
            (
                Approximate(:left, statistic, 1.0; scale=1.0),
                Approximate(:right, statistic, 3.0; scale=1.0),
            ),
            candidates,
        )
        comparison = compare(fit(Xoshiro(9), tradeoff))
        if evidence == :certified
            @test comparison.incomparable == [((:left, 1), (:right, 1))]
        else
            @test comparison.unresolved == [((:left, 1), (:right, 1))]
        end
    end
end
