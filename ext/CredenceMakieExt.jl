module CredenceMakieExt

using Credence: Credence, FitReport, Validation
using Makie: Axis, Figure, axislegend, lines!, xlims!
using MeasureBase: MeasureBase, logdensityof, massof, rootmeasure
using VarNames: VarName

function Credence.priorplot(
    input::Union{FitReport,Validation},
    target::Union{Nothing,Symbol,VarName}=nothing;
    bounds::Tuple{Real,Real},
    xlabel=target === nothing ? "Parameter" : string(target),
)
    lower, upper = bounds
    isfinite(lower) && isfinite(upper) && lower < upper ||
        throw(ArgumentError("plot bounds must be finite and increasing"))
    report = input isa Validation ? Credence._fit_report(input.report) : input
    reference = target === nothing ? nothing : Credence._reference(target)
    figure = Figure(; size=(660, 380))
    axis = Axis(figure[1, 1]; xlabel, ylabel="Prior density")
    grid = range(lower, upper; length=402)[2:(end-1)]
    plotted = false
    for result in report.results
        result.error === nothing || continue
        law = result.measure
        if reference !== nothing
            scope = result.candidate.scope
            Credence._covers(law, scope, reference) || continue
            law = Credence._law_reference(law, scope, reference)
        end
        rootmeasure(law) isa Union{MeasureBase.LebesgueBase,MeasureBase.Lebesgue} &&
        MeasureBase.mspace_elsize(law) == () &&
        isone(massof(law)) || throw(
            ArgumentError("result $(result.id) needs a normalized continuous scalar law"),
        )
        density = map(grid) do x
            value = exp(logdensityof(law, x))
            isfinite(value) ? value : NaN
        end
        label = "$(result.id[end-1]) (start $(last(result.id)))"
        lines!(axis, grid, density; label)
        plotted = true
    end
    plotted || throw(ArgumentError("no fitted density found for the requested target"))
    xlims!(axis, lower, upper)
    axislegend(axis)
    return figure
end

end
