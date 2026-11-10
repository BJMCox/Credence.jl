"""
    Credence

Expert-guided prior elicitation for Julia.
"""
module Credence

using HeterogeneousComputing: GenContext
using MeasureBase
using Random: AbstractRNG
import StatsAPI: fit
using Statistics: mean, std

export Probability, Quantile, Mean, Std, Statistic, Estimate, evaluate
export Hard, Approximate, AtLeast, AtMost, Between, EqualTo
export Candidate, Elicitation, Evaluate, FitResult, FitReport, fit, compare
export Analytic, MonteCarlo, Check, Validation, validate, select, measure

include("statistics.jl")
include("beliefs.jl")
include("fitting.jl")
include("validation.jl")

end
