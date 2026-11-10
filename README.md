# Credence.jl

<a href="https://bjmcox.github.io/Credence.jl/stable/"><img src="/docs/src/assets/logo.svg" align="right" alt="Credence.jl logo" style="height: 140px;"></a>

Expert-guided prior elicitation for Julia.

Credence is an initial prototype. It fits scalar prior families to probability, quantile,
mean, and standard-deviation beliefs. It keeps candidate alternatives, checks hard requirements,
and returns an explicitly selected MeasureBase measure.

Optimization.jl provides optional numerical fitting. Plaice support lets wrapped measures and
named products enter Wren models as joint priors, preserving dependence and support transforms.

Start with the [documentation](docs/src/index.md), [scalar example](examples/scalar.jl),
or [Wren example](examples/wren.jl). The tutorials use those runnable files as their source.
The prototype targets pinned MeasureBase and HeterogeneousComputing development APIs.

Run the scalar fitting example with Julia 1.13:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/scalar.jl
```

## Development

Run the tests from the package directory:

```sh
julia --project=. dev/setup.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Build the documentation:

```sh
julia --project=docs -e 'include("dev/setup.jl"); Pkg.develop(path=".")'
julia --project=docs docs/make.jl
```

Run the Wren integration tests with Julia 1.13 or later and access to the private
Wren repository:

```sh
julia --project=test/integration -e 'using Pkg; Pkg.instantiate()'
julia --project=test/integration test/integration/runtests.jl
```

This prototype does not yet implement the full elicitation workflow. Formula macros, predictive
fitting, tilting, marginal locks, judgment-uncertainty propagation, notebook controls, and FlatPPL
export remain planned. Joint Wren inference is tested with ForwardDiff and Enzyme.
The Enzyme example elides only its read-only prior's defensive copy to avoid
[Enzyme #3527](https://github.com/EnzymeAD/Enzyme.jl/issues/3527), related to
[#3561](https://github.com/EnzymeAD/Enzyme.jl/issues/3561). It uses the pinned Wren
version's internal copy-selection API. Copies of mutable work buffers must remain enabled.
