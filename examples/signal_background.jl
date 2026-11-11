# # Signal and background model
#
# A rare-event experiment searches for a narrow Gaussian signal over an exponential
# background. Five datasets have different exposures and detection efficiencies.
# The hierarchy follows the [BAT.jl example](https://doi.org/10.1007/s42979-021-00626-4),
# with different experimental settings, simulated data, and prior judgments.
# Here both the signal location and detector resolution are uncertain.
#
# ## Parameters and units
#
# Exposure is in years, rates in events/year, and energy in keV. The signal rate `S`
# applies before detection efficiency. The bounds below represent external knowledge
# about possible parameter values. Quantile judgments describe probability within these bounds.

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

# ## Background hierarchy
#
# Each dataset has a background rate `B[i]`. Conditional on `m_B` and `sigma_B`, these
# rates are independent LogNormal variables. The correction to the log location gives
# each rate conditional mean `m_B`. The parameter `sigma_B` is their log standard deviation.

function background_parameters(rng, hyperparameters)
    law = LogNormal(
        log(hyperparameters.m_B) - hyperparameters.sigma_B^2 / 2,
        hyperparameters.sigma_B,
    )
    B = rand(rng, law, length(experiment.exposure))
    return (; hyperparameters..., B)
end
nothing # hide

# ## Counts and event energies
#
# Expected background counts equal exposure times background rate. Expected signal counts
# also include detection efficiency. Their sum is the mean of the observed Poisson count.
# Each event energy follows a mixture of the exponential background and Gaussian signal,
# weighted by their expected counts. The exponential scale `lam` is its mean energy in keV.
# Every event energy enters the likelihood, including energies outside the plotted range.

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

# The likelihood combines the Poisson counts and mixture densities of individual energies.
# `MixtureModel` evaluates in log space to handle small scales and zero signal rates.
# Counts and energy rows must describe the same events in each dataset.
#
# ## Prior families
#
# We compare two joint hyperpriors. One uses scaled Beta distributions for the five positive
# parameters and a truncated Normal for the signal location. The other uses truncated
# LogNormal distributions and a symmetric scaled Beta location distribution.
# Both location families are symmetric about 125 keV. Both joint hyperpriors assume independent
# hyperparameters. The conditional background rates become dependent through their shared
# uncertain mean and log standard deviation.

scaled_beta(bounds, a, b) =
    LocationScale(first(bounds), last(bounds) - first(bounds), Beta(exp(a), exp(b)))

# Family descriptors supply parameter transforms and starting values. A custom builder
# gives the location Beta distribution equal shape parameters, enforcing symmetry.

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

# Each combination lists one family for each parameter. `Independent` forms their product.
