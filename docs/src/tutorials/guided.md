# [Guided elicitation for a rare-event search](@id guided-tutorial)

This Julia REPL session elicits priors for a weak Gaussian line above an exponential
background. It follows the [scientific model](@ref rare-event-model) and reaches the same
final judgments as the [programmatic workflow](@ref scalar-tutorial).

From a Credence checkout, start the example environment with Julia 1.13 or later:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples
```

```@repl guided
using Credence, MeasureBase, Optimization, OptimizationOptimJL, Random
include(joinpath(pkgdir(Credence), "examples", "signal_background.jl"));
include(joinpath(pkgdir(Credence), "examples", "feedback.jl"));
```

## Agree on quantities and hard bounds

Ask: **Which rate and which energy are we discussing? What can independent knowledge rule out?**

| Quantity | Meaning | Illustrative support |
|:--|:--|:--|
| `S` | Signal rate before efficiency, in events/year | `[0, 8]` |
| `mu` | Gaussian line location, in keV | `[122, 128]` |
| `width` | Gaussian signal width, in keV | `[0.8, 3]` |
| `sigma_B` | Dimensionless log variation of background rates | `[0.15, 0.9]` |
| `m_B` | Conditional mean background rate, in events/year | `[0.001, 15]` |
| `lam` | Exponential background energy scale, in keV | `[0.001, 120]` |

The width describes the spread of individual signal energies around `mu`. Its prior
expresses uncertainty about detector resolution. The prior on `mu` instead describes
uncertainty about the line location. The bounds represent illustrative external assumptions.
Every candidate enforces them through its support, independently of solver tolerance.

`S ≤ 8` does not bound the realized Poisson count. A 95% prior upper quantile for `S`
is also a different statement from a hard cap or a posterior upper limit.

Both location families assume symmetry about 125 keV. Confirm this family assumption
separately from the probability question below. All six hyperparameters receive elicited
priors. No background hyperprior remains an implicit default.

## Ask for probability judgments

Ask: **Which signal rate has half your prior probability below it?**
The illustrative answer is **1.2 events/year**.

Ask: **Which signal rate has 95% of your prior probability below it?**
The initial answer is **5 events/year**.

Ask: **What probability do you assign to the line lying between 124 and 126 keV?**
The answer is **90%**.

For each remaining positive quantity, ask the same median and 95% upper-quantile questions.
These judgments could come from calibration studies or previous detector runs.
For `sigma_B`, discuss variation on the log-rate scale before asking for its uncertainty.
Conditional background rates have log standard deviation `sigma_B`, not standard deviation `sigma_B`.

| Quantity | Median | 95% upper quantile | Discrepancy scale |
|:--|--:|--:|--:|
| Signal width, keV | 1.6 | 2.1 | 0.25 keV |
| Background log variation | 0.4 | 0.6 | 0.1 |
| Mean background rate, events/year | 7 | 10.5 | 2 events/year |
| Background energy scale, keV | 55 | 80 | 15 keV |

Pause if the expert cannot judge. An unanswered question does not become zero or a uniform
prior. These 11 illustrative answers concern external knowledge, not the search events.

```@repl guided
initial_beliefs = (
    :S => (
        Approximate(:median, Quantile(0.5), 1.2; scale=1.0),
        Approximate(:upper, Quantile(0.95), 5.0; scale=1.0),
    ),
    :mu => (Approximate(:central, Probability(location_interval...), 0.9; scale=0.1),),
    :width => (
        Approximate(:median, Quantile(0.5), 1.6; scale=0.25),
        Approximate(:upper, Quantile(0.95), 2.1; scale=0.25),
    ),
    :sigma_B => (
        Approximate(:median, Quantile(0.5), 0.4; scale=0.1),
        Approximate(:upper, Quantile(0.95), 0.6; scale=0.1),
    ),
    :m_B => (
        Approximate(:median, Quantile(0.5), 7.0; scale=2.0),
        Approximate(:upper, Quantile(0.95), 10.5; scale=2.0),
    ),
    :lam => (
        Approximate(:median, Quantile(0.5), 55.0; scale=15.0),
        Approximate(:upper, Quantile(0.95), 80.0; scale=15.0),
    ),
)
```

Read the probability statements back to the expert before fitting. Each parameter key owns
its explicitly named beliefs. `Quantile` and `Probability` define their statistics directly.
The [model definition](@ref rare-event-model) declares candidate families separately.
All 11 are approximate targets. The scales
and equal importance weights are explicit fitting preferences.

## Compare candidate shapes

The first family uses scaled Beta priors for all five positive quantities and a truncated
Normal location prior. The second uses truncated LogNormal priors and a symmetric scaled Beta location.
Both have the same hard supports. Product hyperparameters are independent in these families.
The dataset-specific backgrounds remain conditional on shared `m_B` and `sigma_B`.

```@repl guided
candidates = signal_candidates();
initial_problem = Elicitation(initial_beliefs, candidates; combinations=signal_combinations());
initial_report = fit(Xoshiro(42), initial_problem, NelderMead(); maxiters=20000, g_tol=1e-12);
collect(prior_summary(initial_report))
collect(target_summary(initial_report, (:beta_normal, 1)))
```

The summary retains every start, its largest scaled residual, and its solver code or error.
`target_summary` shows all 11 achieved targets for any retained result ID.
Solver success is distinct from validation. Small residual differences do not select a family.

## Discuss implications before revising

Ask: **How much probability do these priors place near the hard rate cap? Does that agree
with your external knowledge?**

```@repl guided
[
    (result.id, evaluate(Probability(6.0, 8.0), marginals(result.measure).S).value) for
    result in initial_report.results
]
mkpath("elicitation-output");
initial_strength = feedback(
    initial_report,
    :S,
    strength_bounds;
    xlabel="Signal rate (events/year)",
    levels=(0.5, 0.95),
);
save("elicitation-output/initial-signal-prior.svg", initial_strength);
```

![Initial signal-rate prior densities and cumulative probabilities.](elicitation-output/initial-signal-prior.svg)

The dashed CDF levels mark the elicited 50% and 95% quantiles. Each curve is one retained
fit, not a mixture weight or a credible band. Density endpoints are omitted because a
continuous Beta density can diverge at a boundary without creating a point mass there.

Translate the rate into expected signal events for the known exposures and efficiencies:

```@repl guided
effective_exposure = sum(experiment.exposure .* experiment.efficiency)
initial_problem[:S, :upper].target * effective_exposure
```

This gives a target for the 95% prior quantile of the *expected signal yield*, which is
linear in `S`. It does not specify a quantile of the realized signal count or total count.
No search observations have entered this feedback.

## Revise the judgment and retain the original

Suppose the expert reviews their external rate assessment and changes the 95% rate
quantile to **3.5 events/year**. Record the reason. This is a revised prior judgment,
not a posterior limit extracted from the search data.

```@repl guided
revision_reason = "The expert revised the upper rate quantile after reviewing external rate knowledge and its expected-yield implication.";
revised_problem = replace(initial_problem, :S => Approximate(:upper, Quantile(0.95), 3.5; scale=1.0));
revised_report = fit(Xoshiro(42), revised_problem, NelderMead(); maxiters=20000, g_tol=1e-12);
collect(prior_summary(revised_report))
collect(target_summary(revised_report, (:beta_normal, 1)))
(original=initial_problem[:S, :upper].target, revised=revised_problem[:S, :upper].target)
```

Review all six marginal priors under the revision:

```@repl guided
revised_strength = feedback(
    revised_report,
    :S,
    strength_bounds;
    xlabel="Signal rate (events/year)",
    levels=(0.5, 0.95),
);
save("elicitation-output/revised-signal-prior.svg", revised_strength);
marginal_figure = hyperprior_feedback(revised_report);
save("elicitation-output/hyperpriors.svg", marginal_figure);
comparison = compare(revised_report);
comparison.incomparable
```

![Revised signal-rate prior densities and cumulative probabilities.](elicitation-output/revised-signal-prior.svg)
![Cumulative probabilities for all six elicited hyperparameters.](elicitation-output/hyperpriors.svg)

Symmetry makes 90% central location probability correspond to CDF levels 5% and 95%
at 124 and 126 keV. The location-support endpoints remain 122 and 128 keV.
Feedback does not add a judgment unless the expert explicitly adopts it.

## Validate the current report and select

Validation belongs to its report. An assessment of the initial fit cannot accept the revision:

```@repl guided
initial_validation = validate(Xoshiro(43), initial_report);
initial_validation.report === revised_report
validation = validate(Xoshiro(43), revised_report);
[[check.status for check in row] for row in validation.checks]
```

These checks are `:target`, because the judgments are approximate. Review the achieved
values and solver codes above. Physical support bounds come from the candidate laws.
Approximate quantile targets can differ slightly from their requested values numerically.

Ask: **Which candidate do you accept, and why?**

For this demonstration, the expert chooses the first Beta/Normal fit. The decision and
rationale are inputs, rather than consequences of a ranking rule.

```@repl guided
selection = select(
    revised_report,
    (:beta_normal, 1);
    validation,
    rationale="The expert accepts the bounded Beta hyperparameters and calibrated Normal location after reviewing both alternatives.",
);
prior = measure(selection);
rng = Xoshiro(44);
hyperparameters = rand(rng, prior)
parameters = background_parameters(rng, hyperparameters);
(S=parameters.S, mu=parameters.mu, width=parameters.width, B=parameters.B)
```

The selected measure is a hyperprior. Drawing `B` conditionally completes the hierarchical
prior and preserves its shared background-rate structure. Use `role=:reference` when
retaining a fit for discussion without accepting it. A new judgment needs a new problem,
fresh validation, and a new explicit decision.

## Inspect the joint prior

This parameter point has six hyperparameters and five conditional background rates.
Draw paired values from the selected prior to inspect their joint meaning:

```@repl guided
joint_figure = joint_feedback(Xoshiro(46), prior);
save("elicitation-output/joint-prior.svg", joint_figure);
```

![Joint signal and hierarchical background prior draws.](elicitation-output/joint-prior.svg)

The left panel shows signal rate and location, coloured by signal width. Their independence
is an explicit family assumption, not a result inferred from marginal judgments.
The right panel shows background rates from the same joint draws. Shared uncertain
hyperparameters make these rates dependent. In particular, their covariance equals the
prior variance of `m_B`, since each conditional mean equals `m_B`.

## Use the scientific likelihood

Only after the prior exercise do we generate a fresh synthetic search dataset. The truth
values below are a reproducibility device. They were not used to fit or revise the prior.
The exposures, efficiencies, and generated observations differ from the original example.

```@repl guided
truth = (S=0.8, mu=125.4, width=1.6, sigma_B=0.35, m_B=7.0, lam=55.0);
truth_parameters = background_parameters(Xoshiro(2026), truth);
data = simulate_spectrum(Xoshiro(2027), truth_parameters);
(counts=data.counts, total=sum(data.counts), energy_rows=length(data.energy))
spectrum_loglikelihood(parameters, data)
predictive_figure = spectrum_feedback(Xoshiro(45), prior, data; ndraws=10_000);
save("elicitation-output/prior-predictive.svg", predictive_figure);
```

![Prior expected spectra and predictive counts compared with synthetic observations.](elicitation-output/prior-predictive.svg)

The run uses 10,000 hierarchical prior predictive draws. Blue curves show expected binned
spectra for the first 200 draws. Black points show synthetic observed counts.
The right panel uses all 10,000 Poisson count draws and marks
the observed total. These are prior predictions, not posterior fits or hard-bound certificates.
The likelihood uses individual energies. Plot bins and display limits do not select events.
Do not reuse these search observations as independent prior-elicitation information.

The programmatic workflow replays the final beliefs, fitting settings, and explicit choice.
Keep the questions, revisions, family assumptions, seeds, and environment with that script.
