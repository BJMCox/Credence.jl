using CairoMakie

function feedback(report)
    figure = Figure(; size=(800, 360))
    density = Axis(figure[1, 1]; xlabel="Offset (°C)", ylabel="Prior density")
    probability = Axis(figure[1, 2]; xlabel="Offset (°C)", ylabel="P(offset ≤ x)")
    points = range(-5, 5; length=301)
    for result in report.results
        result.error === nothing || continue
        prior = result.measure
        label = "$(result.id[1]), start $(result.id[2])"
        lines!(density, points, exp.(MeasureBase.logdensityof.(Ref(prior), points)); label)
        values = [evaluate(Probability(-Inf, x), prior).value for x in points]
        lines!(probability, points, values)
    end
    hlines!(probability, [0.25, 0.75]; color=:gray, linestyle=:dash)
    ylims!(probability, 0, 1)
    Legend(figure[2, 1:2], density; orientation=:horizontal, framevisible=false)
    return figure
end
