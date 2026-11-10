# Contributing to Credence.jl

Bug reports, examples, documentation improvements, and code contributions are welcome.
Use [issues](https://github.com/BJMCox/Credence.jl/issues) to discuss a change or report a bug.
Include a small runnable example, the Julia version, and the expected and observed behavior.

## Development setup

Use Julia 1.13 or later for the examples, documentation, and Wren integration environment.
The core supports Julia 1.10 and later. Run commands from the repository root.

The prototype uses pinned MeasureBase and HeterogeneousComputing development APIs.
Install those pins with the setup script:

```sh
julia --project=. dev/setup.jl
```

## Package tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

CI tests Julia 1.10 on Linux and current Julia on Linux, macOS, and Windows.
Keep tests focused on public behavior and preserve explicit prior selection and validation.

## Examples

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/scalar.jl
```

The scalar example fits and compares prior families. Its source also generates the
corresponding documentation tutorial.

## Documentation

```sh
julia --project=docs -e 'include("dev/setup.jl"); Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The build runs doctests, checks exported API documentation, and fails on warnings.
Generated HTML goes in `docs/build/`. Edit the runnable files in `examples/` to change the
tutorials. Edit other documentation pages in `docs/src/`.

The README and documentation share the transparent logo in `docs/src/assets/logo.svg`.
Keep the favicon in `docs/src/assets/favicon.ico` consistent with the logo.

## Wren integration

The Wren repository is private. These checks require access to its pinned revision:

```sh
julia --project=test/integration -e 'using Pkg; Pkg.instantiate()'
julia --project=test/integration test/integration/runtests.jl
julia --project=test/integration examples/wren.jl
```

The integration tests cover joint Wren inference with ForwardDiff and Enzyme. They run
locally because public CI cannot access the private dependency.

The Enzyme example elides only its read-only prior's defensive copy to avoid
[Enzyme #3527](https://github.com/EnzymeAD/Enzyme.jl/issues/3527), related to
[#3561](https://github.com/EnzymeAD/Enzyme.jl/issues/3561). It uses the pinned Wren
version's internal copy-selection API. Keep copies of mutable work buffers enabled.

## Pull requests

Create a branch and target `main`. Keep each change focused and describe its user-visible
effect. Update examples and documentation when behavior changes.

Run the relevant tests and the documentation build before requesting review. Include the
commands and results in the pull request. Note any private integration checks you could not run.
CI also checks Julia and Markdown formatting with JuliaFormatter.
