# Credence.jl

[![CI](https://github.com/BJMCox/Credence.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BJMCox/Credence.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-guide-blue.svg)](docs/src/index.md) <a href="docs/src/index.md"><img src="/docs/src/assets/logo.svg" align="right" alt="Credence.jl logo" style="height: 140px;"></a>
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Expert-guided prior elicitation for Julia.

Credence fits scalar and joint prior families to judgments about probabilities, quantiles,
means, and standard deviations. The expert compares fitted candidates and records a
selection. Priors use MeasureBase, with fitting reports and validation retained separately.

The prototype supports elicitation from a single expert. Parameter references group beliefs
about scalars, arrays, joint quantities, and derived quantities. Optimization.jl provides
numerical fitting. Plaice connects selected joint priors to Wren models.

## Installation

Credence is not registered. Clone the repository and install its pinned development dependencies:

```sh
git clone https://github.com/BJMCox/Credence.jl.git
cd Credence.jl
julia --project=. dev/setup.jl
```

The core supports Julia 1.10 and later. The example environments require Julia 1.13 or later.

## Quick start

Start Julia with `julia --project=.` from the checkout. Check which of two fixed priors
satisfies a requirement on the median, then select one:

```julia
using Credence, MeasureBase, Random

belief = Hard(:median, Quantile(0.5), Between(-0.1, 0.1))
candidates = (
    Candidate(:normal, _ -> StdNormal(), [0.0]),
    Candidate(:exponential, _ -> StdExponential(), [0.0]),
)
report = fit(Xoshiro(1), Elicitation((belief,), candidates))
checks = validate(Xoshiro(2), report)
selection = select(
    report,
    (:normal, 1);
    validation=checks,
    rationale="The standard Normal satisfies the median requirement.",
)
prior = measure(selection)
rand(Xoshiro(3), prior^5)
```

Here `fit` evaluates the declared parameter points. The
[signal-background example](examples/scalar.jl) compares prior families for six model hyperparameters,
revises a judgment, and examines prior predictions.

## Documentation and support

Start with the [guide](docs/src/index.md), [beliefs and validation](docs/src/validation.md),
or [API reference](docs/src/api.md). The [Wren example](examples/wren.jl) shows a joint prior
and requires access to the private Wren repository.

The [FlatPPL example](examples/flatppl.jl) elicits a joint prior and exports a complete
counting model, with density checks against Julia.

[Open an issue](https://github.com/BJMCox/Credence.jl/issues) for questions, bug reports,
or feature requests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, documentation builds,
and pull requests. [COLLABORATORS.md](COLLABORATORS.md) lists the package author and collaboration links.

Credence uses the [Apache 2.0 license](LICENSE).
