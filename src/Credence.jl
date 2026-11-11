"""
    Credence

Expert-guided prior elicitation for Julia.
"""
module Credence

using HeterogeneousComputing:
    GenContext, get_rng, get_precision, get_compute_unit, allocate_array
using MeasureBase
using LinearAlgebra
import InverseFunctions: inverse
import ChangesOfVariables: with_logabsdet_jacobian
using Random: Random, AbstractRNG, rand, Xoshiro
using VarNames: VarNames, VarName, getvalue, getsym, getoptic, subsumes
import StatsAPI: fit
import Base: unlock
using Statistics: mean, std, cov, cor, var, quantile

export Probability, Quantile, Mean, Std, Statistic, Estimate, evaluate
export Hard, Approximate, AtLeast, AtMost, Between, EqualTo
export Candidate, Elicitation, Evaluate, FitResult, FitReport, fit, compare
export Analytic, MonteCarlo, Check, Validation, validate, select, measure
export Joint, Derived, Correlation, Covariance, CorrelationMatrix, CovarianceMatrix
export NormalFamily, LogNormalFamily, ScaledBetaFamily, TruncatedFamily, IIDFamily
export MvNormalFamily, MvLogNormalFamily, Independent, AllCombinations, unlock
export diagnostics, flatppl
export fit_summary, belief_summary

include("statistics.jl")
include("beliefs.jl")
include("targets.jl")
include("fitting.jl")
include("families.jl")
include("composition.jl")
include("recordmap.jl")
include("evaluation.jl")
include("validation.jl")
include("reporting.jl")
include("flatppl.jl")

end
