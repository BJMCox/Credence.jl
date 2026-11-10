using Test
using Credence
using MeasureBase: MeasureBase
using Distributions: Distributions, Normal, MvNormal, LogNormal
using PDMats
using Optimization, OptimizationOptimJL
using Plaice
using Random: Xoshiro
using LinearAlgebra
using ADTypes: AutoForwardDiff
using ForwardDiff
import Enzyme
using LogDensityProblems: LogDensityProblems
using Wren: @wren, @observe, VecModel, FromPrior, nuts, AutoEnzyme, Reverse
import WrenCore
using VarNames: @vn

const covariance = [1.0 0.4; 0.4 0.8]

function joint_prior(p)
    MeasureBase.productmeasure((
        beta=MeasureBase.asmeasure(MvNormal(p[1:2], covariance)),
        sigma=MeasureBase.asmeasure(LogNormal(p[3], 0.4)),
    ))
end

@wren function scalar_prior(prior)
    theta ~ prior
end

@testset "Scalar selected prior with Enzyme" begin
    candidate = Candidate(:normal, p -> MeasureBase.asmeasure(Normal(p[1], 0.8)), [0.2])
    problem = Elicitation([Hard(:mean, Mean(), EqualTo(0.2))], [candidate])
    report = fit(Xoshiro(786), problem)
    validation = validate(Xoshiro(787), report)
    selection =
        select(report, (:normal, 1); validation, rationale="Scalar integration check")
    vm = VecModel(
        scalar_prior(measure(selection)),
        AutoEnzyme(; mode=Reverse);
        latent_source=FromPrior(Xoshiro(788)),
    )
    value, gradient = LogDensityProblems.logdensity_and_gradient(vm, [0.4])
    @test value ≈ Distributions.logpdf(Normal(0.2, 0.8), 0.4)
    @test only(gradient) ≈ -(0.4-0.2) / 0.8^2
end

@wren function regression(prior, y)
    theta ~ prior
    @observe y ~ Normal(theta.beta[1] + 2 * theta.beta[2], theta.sigma)
end

@testset "Selected prior through Plaice and Wren" begin
    coefficient(i) = Statistic(p -> Distributions.mean(MeasureBase.marginals(p).beta)[i])
    log_scale =
        Statistic(p -> log(evaluate(Quantile(0.5), MeasureBase.marginals(p).sigma).value))
    problem = Elicitation(
        (
            Approximate(:first_coefficient, coefficient(1), 0.2; scale=1.0, importance=1.0),
            Approximate(
                :second_coefficient,
                coefficient(2),
                -0.1;
                scale=1.0,
                importance=1.0,
            ),
            Approximate(:log_scale, log_scale, -0.3; scale=1.0, importance=1.0),
        ),
        [Candidate(:joint, joint_prior, [0.0, 0.0, 0.0])],
    )
    report = fit(Xoshiro(781), problem, NelderMead(); maxiters=3000, g_tol=1e-14)
    fitted = collect(only(report.results).parameters)
    @test fitted ≈ [0.2, -0.1, -0.3] atol=1e-5
    validation = validate(Xoshiro(782), report)
    selection =
        select(report, (:joint, 1); validation, rationale="Explicit prototype choice")
    prior = measure(selection)

    original = (beta=[0.4, -0.2], sigma=0.8)
    flat = [0.4, -0.2, 0.8]
    z = [0.4, -0.2, log(0.8)]
    @test Plaice.to_vec(prior)(original) == flat
    @test Plaice.from_vec(prior)(flat) == original
    @test Plaice.to_unconstrained_vec(prior)(original) ≈ z
    restored, inverse_logjac =
        Plaice.with_logabsdet_jacobian(Plaice.from_unconstrained_vec(prior), z)
    @test restored.beta ≈ original.beta
    @test restored.sigma ≈ original.sigma
    @test inverse_logjac ≈ z[3]
    @test map(optic -> optic(original), Plaice.optic_vec(prior)) == flat
    @test Plaice.vec_length(prior) == Plaice.unconstrained_vec_length(prior) == 3

    model = regression(prior, 0.4)
    draw = rand(Xoshiro(783), model)[@vn(theta)]
    @test draw.sigma > 0
    @test isfinite(MeasureBase.logdensityof(prior, draw))
    vm = VecModel(model, AutoForwardDiff(); latent_source=FromPrior(Xoshiro(784)))
    value, gradient = LogDensityProblems.logdensity_and_gradient(vm, z)

    # The lognormal density and inverse chart Jacobian cancel to a Normal in log(sigma).
    residual = 0.4 - (z[1] + 2z[2])
    expected =
        Distributions.logpdf(MvNormal(fitted[1:2], covariance), z[1:2]) +
        Distributions.logpdf(Normal(fitted[3], 0.4), z[3]) +
        Distributions.logpdf(Normal(z[1] + 2z[2], exp(z[3])), 0.4)
    expected_gradient = vcat(
        -(covariance \ (z[1:2] - fitted[1:2])) + residual / exp(2z[3]) * [1, 2],
        -(z[3]-fitted[3]) / 0.4^2 - 1 + residual^2 / exp(2z[3]),
    )
    @test LogDensityProblems.dimension(vm) == 3
    @test value ≈ expected
    @test gradient ≈ expected_gradient
    @test LogDensityProblems.logdensity(vm, z) ≈ value

    copies = filter(!=(:prior), WrenCore._copied_args(model))
    readonly_model = WrenCore._with_copied_args(model, copies)
    enzyme = VecModel(
        readonly_model,
        AutoEnzyme(; mode=Reverse);
        latent_source=FromPrior(Xoshiro(789)),
        reduce_copies=false,
    )
    for point in (z, [0.0, 0.0, 0.0], [1.1, -0.4, 0.2])
        fd_value, fd_gradient = LogDensityProblems.logdensity_and_gradient(vm, point)
        enz_value, enz_gradient = LogDensityProblems.logdensity_and_gradient(enzyme, point)
        @test enz_value ≈ fd_value
        @test enz_gradient ≈ fd_gradient
    end
    chain = nuts(
        Xoshiro(785),
        readonly_model,
        32;
        adtype=AutoEnzyme(; mode=Reverse),
        reduce_copies=false,
        progress=false,
    )
    @test size(chain) == (32, 1)
end

@wren function buffered_regression(prior, buffer, y)
    theta ~ prior
    for i in eachindex(buffer)
        buffer[i] = theta.beta[1]
    end
    @observe y ~ Normal(sum(buffer), theta.sigma)
end

@testset "Read-only prior elision keeps mutable buffers protected" begin
    prior = joint_prior([0.2, -0.1, -0.3])
    caller_buffer = zeros(2)
    model = buffered_regression(prior, caller_buffer, 0.4)
    copies = filter(!=(:prior), WrenCore._copied_args(model))
    model = WrenCore._with_copied_args(model, copies)
    vm = VecModel(
        model,
        AutoEnzyme(; mode=Reverse);
        latent_source=FromPrior(Xoshiro(790)),
        reduce_copies=false,
    )
    before = copy(model.args.buffer)
    z = [0.4, -0.2, log(0.8)]
    residual = 0.4-2z[1]
    expected_gradient = vcat(
        -(covariance \ (z[1:2]-[0.2, -0.1])) + residual/exp(2z[3])*[2, 0],
        -(z[3]+0.3)/0.4^2 - 1 + residual^2/exp(2z[3]),
    )
    _, gradient = LogDensityProblems.logdensity_and_gradient(vm, z)
    @test gradient ≈ expected_gradient
    @test model.args.buffer == before
    @test caller_buffer == zeros(2)
end
