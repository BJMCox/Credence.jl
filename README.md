# Credence.jl

Expert-guided prior elicitation for Julia.

Credence is in early development. The package currently provides its initial structure.
The planned workflow records beliefs, compares candidate priors, validates their implications,
and retains an explicit expert selection.

## Development

Run the tests from the package directory:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Build the documentation:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```
