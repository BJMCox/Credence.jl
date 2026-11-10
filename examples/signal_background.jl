# # [A rare-event signal and background model](@id rare-event-model)
#
# This synthetic example follows the hierarchical structure of the
# [BAT.jl extended example](https://doi.org/10.1007/s42979-021-00626-4).
# Exposures, efficiencies, rates, energies, and prior judgments differ from the original.
# Both the line location and detector resolution are uncertain here.
#
# ## Experimental design and units
#
# Exposure is measured in years, rates in events/year, and energies in keV.
# `S` is a signal rate before efficiency, not a peak height or an event energy.
# The support bounds below are illustrative external assumptions, not limits inferred
# from the observed events. A 95% prior upper quantile is a separate judgment.

using Credence, MeasureBase
using Distributions:
    Beta, Exponential, LocationScale, LogNormal, MixtureModel, Normal, Poisson, logpdf

const experiment =
    (exposure=(1.8, 1.5, 1.1, 0.8, 0.5), efficiency=(0.52, 0.61, 0.72, 0.81, 0.90))
const prior_bounds = (
    S=(0.0, 8.0),
    width=(0.8, 3.0),
    sigma_B=(0.15, 0.9),
    m_B=(0.001, 15.0),
    lam=(0.001, 120.0),
    mu=(122.0, 128.0),
)
const strength_bounds = prior_bounds.S
const location_bounds = prior_bounds.mu
const location_interval = (124.0, 126.0)
const location_center = sum(location_bounds) / 2
nothing # hide

# ## Hierarchical background and event likelihood
#
# For dataset i, the expected yields are exposure times background rate, and exposure
# times efficiency times signal rate. Conditional background rates share a lognormal law.
# Its location correction makes the conditional arithmetic mean equal to `m_B`.
# `sigma_B` is dimensionless. `lam` is an exponential energy scale, not a decay rate:
# Julia's `Exponential(lam)` has mean `lam`.

function background_parameters(rng, hyperparameters)
    law = LogNormal(
        log(hyperparameters.m_B) - hyperparameters.sigma_B^2 / 2,
        hyperparameters.sigma_B,
    )
    B = rand(rng, law, length(experiment.exposure))
    return (; hyperparameters..., B)
end
nothing # hide

# Counts follow a Poisson law. Each event energy follows a normalized mixture whose
# weights are the expected yields. Retain every simulated energy. No analysis-window
# selection is applied. A finite plot window does not truncate this likelihood.

function expected_yields(parameters)
    background = experiment.exposure .* parameters.B
    signal = experiment.exposure .* experiment.efficiency .* parameters.S
    return background, signal
end

function energy_law(parameters, background, signal)
    return MixtureModel(
        [Exponential(parameters.lam), Normal(parameters.mu, parameters.width)],
        [background, signal] ./ (background + signal),
    )
end

function simulate_spectrum(rng, parameters)
    background, signal = expected_yields(parameters)
    counts, dataset, energy = Int[], Int[], Float64[]
    for i in eachindex(background)
        n = rand(rng, Poisson(background[i] + signal[i]))
        push!(counts, n)
        append!(dataset, fill(i, n))
        append!(energy, rand(rng, energy_law(parameters, background[i], signal[i]), n))
    end
    return (; counts, dataset, energy)
end

function spectrum_loglikelihood(parameters, data)
    background, signal = expected_yields(parameters)
    laws = energy_law.(Ref(parameters), background, signal)
    counts = sum(logpdf.(Poisson.(background .+ signal), data.counts))
    energies =
        sum(logpdf(laws[i], energy) for (i, energy) in zip(data.dataset, data.energy))
    return counts + energies
end
nothing # hide

# `MixtureModel` evaluates log densities without first adding ordinary component densities.
# This keeps the likelihood usable for small background scales and for `S = 0`.
# Counts and the number of energy rows per dataset must describe the same observations.
#
# ## Candidate hyperpriors
#
# Compare scaled Beta positive quantities with a truncated Normal location, and
# truncated LogNormal quantities with a symmetric scaled Beta location. Both location
# families assume symmetry about 125 keV. This is separate from probability judgments.
# All six hyperparameters are elicited together. Their independence is an explicit
# family assumption. Conditional backgrounds share `m_B` and `sigma_B`, so the five
# background rates are dependent after those hyperparameters are integrated out.

scaled_beta(bounds, a, b) =
    LocationScale(first(bounds), last(bounds) - first(bounds), Beta(exp(a), exp(b)))

# Standard descriptors own their parameter transforms and starts. The location Beta
# uses the custom form to impose symmetry explicitly through one shared shape parameter.

function signal_candidates()
    return (
        :S => (
            :beta => ScaledBetaFamily(prior_bounds.S...),
            :lognormal => TruncatedFamily(LogNormal, prior_bounds.S...),
        ),
        :width => (
            :beta => ScaledBetaFamily(prior_bounds.width...),
            :lognormal => TruncatedFamily(LogNormal, prior_bounds.width...),
        ),
        :sigma_B => (
            :beta => ScaledBetaFamily(prior_bounds.sigma_B...),
            :lognormal => TruncatedFamily(LogNormal, prior_bounds.sigma_B...),
        ),
        :m_B => (
            :beta => ScaledBetaFamily(prior_bounds.m_B...),
            :lognormal => TruncatedFamily(LogNormal, prior_bounds.m_B...),
        ),
        :lam => (
            :beta => ScaledBetaFamily(prior_bounds.lam...),
            :lognormal => TruncatedFamily(LogNormal, prior_bounds.lam...),
        ),
        :mu => (
            :normal => TruncatedFamily(
                NormalFamily(location=location_center),
                location_bounds...,
            ),
            Candidate(
                :beta,
                p -> asmeasure(scaled_beta(location_bounds, p.log_shape, p.log_shape)),
                ((log_shape=log(12.0),), (log_shape=log(8.0),)),
            ),
        ),
    )
end

function signal_combinations()
    return (
        :beta_normal => Independent(
            :S => :beta,
            :width => :beta,
            :sigma_B => :beta,
            :m_B => :beta,
            :lam => :beta,
            :mu => :normal,
        ),
        :lognormal_beta => Independent(
            :S => :lognormal,
            :width => :lognormal,
            :sigma_B => :lognormal,
            :m_B => :lognormal,
            :lam => :lognormal,
            :mu => :beta,
        ),
    )
end
nothing # hide

# These two combinations explicitly declare independent hyperparameters.
# They avoid silently enumerating every possible mix of marginal families.
# The selected product remains a native MeasureBase measure.

function target_summary(report, id)
    result = only(filter(r -> r.id == id, report.results))
    return map(report.problem.beliefs) do belief
        achieved = evaluate(belief.statistic, result.measure).value
        (; id=belief.id, target=belief.target, achieved, scale=belief.scale)
    end
end

function prior_summary(report)
    return map(report.results) do result
        if result.error !== nothing
            return (; id=result.id, error=result.error)
        end
        rows = target_summary(report, result.id)
        residual = maximum(abs(row.achieved - row.target) / row.scale for row in rows)
        return (;
            id=result.id,
            max_scaled_residual=residual,
            solver=string(result.solver.retcode),
        )
    end
end
nothing # hide
