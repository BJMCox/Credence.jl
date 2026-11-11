module FlatPPLExpressions
using Distributions: Normal, LogNormal, Beta, MvNormal, LocationScale, truncated
using MeasureBase: asmeasure, productmeasure
joint(; laws...) = productmeasure(map(asmeasure, (; laws...)))
joint(laws...) = productmeasure(map(asmeasure, laws))
locscale(law, location, scale) = LocationScale(location, scale, law)
interval(lower, upper) = (lower, upper)
truncate(law, bounds) = truncated(law, bounds...)
normalize(law) = law
rowstack(rows) = reduce(vcat, permutedims.(rows))
iid(law, n::Integer) = asmeasure(law)^n
iid(law, dims::AbstractVector) = asmeasure(law)^Tuple(dims)
evaluate(source) = asmeasure(Core.eval(@__MODULE__, Meta.parse(source)))
end

@testset "FlatPPL export preserves prior densities" begin
    prior = productmeasure((
        S=asmeasure(LocationScale(0.0, 8.0, Beta(2.0, 3.0))),
        mu=asmeasure(truncated(Normal(125.0, 0.6), 122.0, 128.0)),
        beta=asmeasure(MvNormal([0.2, -0.1], [1.0 0.4; 0.4 0.8])),
        sigma=asmeasure(LogNormal(-0.3, 0.4)),
    ))
    source = flatppl(prior)
    @test occursin('\n', source)
    exported = FlatPPLExpressions.evaluate(source)
    for point in (
        (S=1.2, mu=125.0, beta=[0.2, -0.1], sigma=exp(-0.3)),
        (S=7.5, mu=122.1, beta=[-1.0, 0.9], sigma=2.0),
    )
        @test logdensityof(exported, point) ≈ logdensityof(prior, point)
    end
    independent = productmeasure([asmeasure(Normal(1.0, 0.5)), asmeasure(Beta(2.0, 4.0))])
    @test logdensityof(FlatPPLExpressions.evaluate(flatppl(independent)), [0.3, 0.6]) ≈
          logdensityof(independent, [0.3, 0.6])
    matrix_prior = StdNormal()^(2, 3)
    point = [0.2 -0.4 0.3; 1.0 0.5 -0.2]
    @test logdensityof(FlatPPLExpressions.evaluate(flatppl(matrix_prior)), point) ≈
          logdensityof(matrix_prior, point)

    belief = Hard(:median, Quantile(0.5), Between(-0.1, 0.1))
    candidate = Candidate(:normal, _ -> StdNormal(), [0.0])
    report = fit(Xoshiro(52), Elicitation((belief,), (candidate,)))
    validation = validate(Xoshiro(53), report)
    selected = select(report, (:normal, 1); validation, rationale="Centered prior")
    @test logdensityof(FlatPPLExpressions.evaluate(flatppl(selected)), 0.7) ≈
          logdensityof(measure(selected), 0.7)
end
