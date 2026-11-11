# Changelog

## Unreleased

  - Add continuous compound shrinkage priors with joint latent scales, fitting, saved sessions, and noncentred Plaice coordinates.
  - Add positive, heavy-tailed, simplex, and multivariate Student-t families with fixed-parameter options.
  - Evaluate Dirichlet and Student-t marginals and component quantiles analytically.
  - Elicit predictive observations through conditional measures or explicit RNG simulators.
  - Save and restore fitted sessions, revisions, selections, and callback provenance as versioned TOML.
  - Preserve sampling plans during validation and compare candidates with simultaneous uncertainty bounds.
  - Add concise report displays, summary rows, and cross-belief consistency diagnostics.
  - Accept a single named parameter record as a candidate start.
  - Export supported priors as formatted FlatPPL expressions, with a standalone counting-model example.
  - Add named parameter beliefs, standard family adapters, array judgments, and explicit joint composition.
  - Support shared sampled evaluation, partial bounds, and fixed marginal selections.
  - Add the initial fitting, comparison, validation, and explicit-selection workflow.
  - Add optional Optimization and Plaice integrations, with scalar and Wren examples.
  - Work around Enzyme's copy rule for the read-only joint-prior example.
