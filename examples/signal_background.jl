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
    Beta,
    Exponential,
    LocationScale,
    LogNormal,
    MixtureModel,
    Normal,
    Poisson,
    logpdf,
    truncated

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
bounded_lognormal(bounds, m, s) = truncated(LogNormal(m, exp(s)), bounds...)

function joint_hyperprior(p, positive_law, location_law)
    fields = (:S, :width, :sigma_B, :m_B, :lam)
    laws = ntuple(length(fields)) do i
        bounds = getproperty(prior_bounds, fields[i])
        asmeasure(positive_law(bounds, p[2i-1], p[2i]))
    end
    positive = NamedTuple{fields}(laws)
    return productmeasure((; positive..., mu=asmeasure(location_law(p[end]))))
end

# Starts contain two parameters per positive quantity in the order above, followed
# by the location shape parameter. They guide the solver, rather than adding judgments.

function signal_candidates()
    beta_start = log.([1.2, 5.0, 6.6, 11.55, 3.83, 7.67, 6.07, 6.93, 6.82, 8.06, 0.6])
    lognormal_start = [
        log(1.2),
        log(0.7),
        log(1.58),
        log(0.16),
        log(0.39),
        log(0.25),
        log(6.7),
        log(0.28),
        log(53.0),
        log(0.27),
        log(12.0),
    ]
    return (
        Candidate(
            :beta_normal,
            p -> joint_hyperprior(
                p,
                scaled_beta,
                s -> truncated(Normal(location_center, exp(s)), location_bounds...),
            ),
            (beta_start, beta_start .+ 0.1),
        ),
        Candidate(
            :lognormal_beta,
            p -> joint_hyperprior(
                p,
                bounded_lognormal,
                s -> scaled_beta(location_bounds, s, s),
            ),
            (lognormal_start, lognormal_start .+ 0.1),
        ),
    )
end
nothing # hide

# ## Map expert answers to model quantities
#
# Each row declares a marginal statistic, an answer, and a discrepancy scale.
# Median and upper-quantile targets summarize external calibration knowledge.
# These scales and equal importance weights are explicit fitting preferences.
# `S_q95` is an approximate prior quantile target, not a posterior upper limit.

function signal_beliefs(answers)
    specifications = (
        (:S_median, :S, Quantile(0.5), 1.0),
        (:S_q95, :S, Quantile(0.95), 1.0),
        (:mu_probability, :mu, Probability(location_interval...), 0.1),
        (:width_median, :width, Quantile(0.5), 0.25),
        (:width_q95, :width, Quantile(0.95), 0.25),
        (:sigma_B_median, :sigma_B, Quantile(0.5), 0.1),
        (:sigma_B_q95, :sigma_B, Quantile(0.95), 0.1),
        (:m_B_median, :m_B, Quantile(0.5), 2.0),
        (:m_B_q95, :m_B, Quantile(0.95), 2.0),
        (:lam_median, :lam, Quantile(0.5), 15.0),
        (:lam_q95, :lam, Quantile(0.95), 15.0),
    )
    return map(specifications) do (id, field, statistic, scale)
        marginal =
            Statistic(p -> evaluate(statistic, getproperty(marginals(p), field)).value)
        Approximate(id, marginal, getproperty(answers, id); scale, importance=1.0)
    end
end

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
