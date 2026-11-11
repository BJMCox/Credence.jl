using Optimization, OptimizationOptimJL

@testset "Fit and belief summaries retain usable results" begin
    beliefs = (
        Approximate(:location, Mean(), 0.5; scale=2.0),
        Hard(:spread, Std(), Between(0.5, 1.5)),
    )
    candidates = (
        Candidate(:normal, p -> asmeasure(Normal(p[1], 1.0)), [0.0]),
        Candidate(:failure, _ -> error("builder failure"), [0.0]),
    )
    problem = Elicitation(beliefs, candidates)
    report = fit(Xoshiro(72), problem)
    validation = validate(Xoshiro(73), report)
    fits = Credence.fit_summary(report)
    @test fits[1] == (id=(:normal, 1), parameters=(0.0,), status=:evaluated, error=nothing)
    @test fits[2].status == :failed && fits[2].error === report.results[2].error
    optimized = fit(
        Xoshiro(74),
        Elicitation((first(beliefs),), (first(candidates),)),
        NelderMead();
        maxiters=1000,
    )
    @test Credence.fit_summary(optimized)[1].status == optimized.results[1].solver.retcode
    rows = Credence.belief_summary(validation; result_id=(:normal, 1))
    @test rows[1] == (
        result_id=(:normal, 1),
        belief_id=:location,
        target=0.5,
        value=0.0,
        lower=0.0,
        upper=0.0,
        evidence=:analytic,
        status=:target,
        discrepancy=0.25,
        error=nothing,
    )
    @test rows[2].target == (0.5, 1.5) &&
          rows[2].value == 1.0 &&
          rows[2].status == :satisfied
    failed = Credence.belief_summary(validation; result_id=(:failure, 1))
    @test all(
        row -> row.status == :undefined && row.value === nothing && row.error !== nothing,
        failed,
    )
    selection =
        select(report, (:normal, 1); validation, rationale="Reviewed mean and spread")
    @test all((problem, report, report.results[1], validation, selection)) do value
        text = repr(MIME"text/plain"(), value)
        length(text) < 200 && !occursin("var\"", text)
    end
end
