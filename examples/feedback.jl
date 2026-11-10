using CairoMakie
using Distributions: cdf

function feedback(report, field, bounds; xlabel, levels)
    figure = Figure(; size=(800, 360))
    density = Axis(figure[1, 1]; xlabel, ylabel="Prior density")
    probability = Axis(figure[1, 2]; xlabel, ylabel="Prior probability below x")
    points = range(bounds...; length=303)
    for result in report.results
        result.error === nothing || continue
        prior = getproperty(marginals(result.measure), field)
        label = "$(result.id[1]), start $(result.id[2])"
        interior = points[2:(end-1)]
        lines!(
            density,
            interior,
            exp.(MeasureBase.logdensityof.(Ref(prior), interior));
            label,
        )
        values = [evaluate(Probability(-Inf, x), prior).value for x in points]
        lines!(probability, points, values)
    end
    hlines!(probability, collect(levels); color=:gray, linestyle=:dash)
    ylims!(probability, 0, 1)
    Legend(figure[2, 1:2], density; orientation=:horizontal, framevisible=false)
    return figure
end

function hyperprior_feedback(report)
    figure = Figure(; size=(900, 580))
    axes = Axis[]
    labels = (
        "Signal rate (events/year)",
        "Signal width (keV)",
        "Background log variation",
        "Mean background rate (events/year)",
        "Background energy scale (keV)",
        "Line energy (keV)",
    )
    for (i, field) in enumerate(keys(prior_bounds))
        axis = Axis(figure[(i-1)÷3+1, (i-1)%3+1]; xlabel=labels[i], ylabel="Prior CDF")
        push!(axes, axis)
        points = range(getproperty(prior_bounds, field)...; length=201)
        for result in report.results
            result.error === nothing || continue
            law = getproperty(marginals(result.measure), field)
            values = [evaluate(Probability(-Inf, x), law).value for x in points]
            lines!(axis, points, values; label="$(result.id[1]), start $(result.id[2])")
        end
        ylims!(axis, 0, 1)
    end
    Legend(figure[3, 1:3], first(axes); orientation=:horizontal, framevisible=false)
    return figure
end

function joint_feedback(rng, prior; ndraws=1000)
    draws = [background_parameters(rng, rand(rng, prior)) for _ in 1:ndraws]
    figure = Figure(; size=(900, 380))
    signal =
        Axis(figure[1, 1]; xlabel="Signal rate (events/year)", ylabel="Line energy (keV)")
    background = Axis(
        figure[1, 3];
        xlabel="Background rate 1 (events/year)",
        ylabel="Background rate 2 (events/year)",
    )
    points = scatter!(
        signal,
        [p.S for p in draws],
        [p.mu for p in draws];
        color=[p.width for p in draws],
        markersize=4,
    )
    Colorbar(figure[1, 2], points; label="Signal width (keV)")
    scatter!(
        background,
        [p.B[1] for p in draws],
        [p.B[2] for p in draws];
        color=(:steelblue, 0.4),
        markersize=4,
    )
    return figure
end

function spectrum_feedback(rng, prior, data; ndraws=200)
    lower = min(0.0, 5floor(minimum(data.energy) / 5))
    upper = max(200.0, 5ceil(maximum(data.energy) / 5) + 5)
    edges = collect(lower:5.0:upper)
    centres = (edges[1:(end-1)] .+ edges[2:end]) ./ 2
    observed =
        [count(e -> edges[i] <= e < edges[i+1], data.energy) for i in eachindex(centres)]
    figure = Figure(; size=(900, 360))
    spectrum = Axis(figure[1, 1]; xlabel="Energy (keV)", ylabel="Events per 5 keV bin")
    totals = Axis(figure[1, 2]; xlabel="Total events", ylabel="Prior predictive draws")
    draws = Int[]
    for _ in 1:ndraws
        parameters = background_parameters(rng, rand(rng, prior))
        background, signal = expected_yields(parameters)
        background_mass = diff(cdf.(Ref(Exponential(parameters.lam)), edges))
        signal_mass = diff(cdf.(Ref(Normal(parameters.mu, parameters.width)), edges))
        expected = sum(background) .* background_mass .+ sum(signal) .* signal_mass
        lines!(spectrum, centres, expected; color=(:steelblue, 0.1))
        push!(draws, sum(rand.(Ref(rng), Poisson.(background .+ signal))))
    end
    scatter!(spectrum, centres, observed; color=:black, markersize=5)
    hist!(totals, draws; bins=20, color=:steelblue)
    vlines!(totals, [sum(data.counts)]; color=:black)
    return figure
end
