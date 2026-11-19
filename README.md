# Credence.jl

[![CI](https://github.com/BJMCox/Credence.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BJMCox/Credence.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://bjmcox.github.io/Credence.jl/) <a href="https://bjmcox.github.io/Credence.jl/"><img src="/docs/src/assets/logo.svg" align="right" alt="Credence.jl logo" style="height: 140px;"></a>
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Expert-guided prior elicitation for Julia.

Credence fits scalar and joint prior families to judgments about probabilities, quantiles,
means, and standard deviations. The expert compares fitted candidates and records a
selection. Priors use MeasureBase, with fitting reports and validation retained separately.

Credence supports elicitation from a single expert. Parameter references group beliefs
about scalars, arrays, joint quantities, and derived quantities. Optimization.jl provides
numerical fitting. Plaice connects selected joint priors to Wren models.

## Installation

Install Credence from the Julia REPL with Julia 1.13.

```julia
using Pkg
Pkg.add(url="https://github.com/BJMCox/Credence.jl.git")
```

## Quick start

Add Distributions to use the built-in Normal and Exponential families. Check which
candidate meets a requirement on the prior median, then select it.

```julia
Pkg.add("Distributions")
using Credence, Distributions, Random

beliefs = (:x => (Hard(:median, Quantile(0.5), Between(-0.1, 0.1)),),)
candidates = (
    :x => (
        :normal => NormalFamily(location=0.0, scale=1.0),
        :exponential => ExponentialFamily(scale=1.0),
    ),
)
report = fit(Xoshiro(1), Elicitation(beliefs, candidates))
checks = validate(Xoshiro(2), report)
selection = select(
    report,
    :x,
    (:normal, 1);
    validation=checks,
    rationale="The standard Normal satisfies the median requirement.",
)
prior = measure(selection)
rand(Xoshiro(3), prior^5)
```

## Documentation and support

Read the [guide](https://bjmcox.github.io/Credence.jl/),
[worked examples](https://bjmcox.github.io/Credence.jl/dev/#Examples-and-guides),
and [API reference](https://bjmcox.github.io/Credence.jl/dev/api/).

[Open an issue](https://github.com/BJMCox/Credence.jl/issues) for questions, bug reports,
or feature requests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, documentation builds,
and pull requests. [COLLABORATORS.md](COLLABORATORS.md) lists the package author and collaboration links.

Credence uses the [Apache 2.0 license](LICENSE).
