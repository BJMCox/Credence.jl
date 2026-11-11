include("fixtures.jl")
function compound_coordinates(point)
    auxiliary = Base.structdiff(point, (coefficients=nothing,))
    parts = map(keys(auxiliary)) do name
        value = getproperty(auxiliary, name)
        name == :allocation && return value[1:(end-1)]
        value isa Number && return [value]
        vec(value)
    end
    vcat(vec(point.coefficients), parts...)
end

@testset "Compound prior charts and saved sessions" begin
    for (index, family) in enumerate(compound_cases())
        problem = Elicitation((), (:theta => (:compound => family,),))
        report = fit(Xoshiro(2100+index), problem)
        prior = only(report.results).measure
        from = Plaice.from_unconstrained_vec(prior)
        z = range(0.1, 0.5; length=Plaice.unconstrained_vec_length(prior)) |> collect
        point, correction = with_logabsdet_jacobian(from, z)
        @test Plaice.to_unconstrained_vec(prior)(point) ≈ z
        jacobian = ForwardDiff.jacobian(v -> compound_coordinates(from(v)), z)
        @test correction ≈ first(logabsdet(jacobian))
        density(v) = begin
            value, ladj = with_logabsdet_jacobian(from, v)
            logdensityof(prior, value) + ladj
        end
        step = cbrt(eps(Float64))
        numerical = map(eachindex(z)) do i
            delta = zeros(length(z))
            delta[i] = step
            (density(z+delta) - density(z-delta)) / (2step)
        end
        @test ForwardDiff.gradient(density, z) ≈ numerical rtol=sqrt(step) atol=sqrt(step)
        validation = validate(Xoshiro(2200+index), report)
        selection = select(
            report,
            only(report.results).id;
            validation,
            rationale="Reviewed compound prior",
        )
        mktempdir() do directory
            path = joinpath(directory, "compound.toml")
            save_session(path, problem; report, validation, selections=(selection,))
            restored = load_session(path)
            @test logdensityof(measure(only(restored.selections)), point) ≈
                  logdensityof(prior, point)
        end
    end
end

@testset "Compound null boundaries and simplex charts" begin
    family = DirichletLaplaceFamily(2; scale=0.4)
    prior =
        only(fit(Xoshiro(2300), Elicitation((), (:theta => (:law => family,),))).results).measure
    point = (coefficients=[0.1, -0.2], global_scale=1.0, allocation=[0.3, 0.7])
    reversed = (
        allocation=point.allocation,
        coefficients=point.coefficients,
        global_scale=point.global_scale,
    )
    @test logdensityof(prior, point) == logdensityof(prior, reversed)
    @test logdensityof(prior, merge(point, (global_scale=0.0,))) == -Inf
    @test logdensityof(prior, merge(point, (allocation=[0.0, 1.0],))) == -Inf
    scalar = R2D2Family(
        1;
        noise_scale=1.0,
        predictor_scales=[1.0],
        mean_r2=0.4,
        precision_r2=5.0,
    )
    law =
        only(fit(Xoshiro(2301), Elicitation((), (:theta => (:law => scalar,),))).results).measure
    from = Plaice.from_unconstrained_vec(law)
    z = [0.2, 0.3]
    @test Plaice.to_unconstrained_vec(law)(from(z)) ≈ z
    @test isfinite(logdensityof(law, from(z)))
end

@testset "Heavy-tail moments remain undefined under sampled validation" begin
    beliefs =
        (@vn(theta.coefficients) => (Approximate(:mean, Mean(), zeros(2); scale=1.0),),)
    problem = Elicitation(beliefs, (:theta => (:law => HorseshoeFamily(2; scale=0.3),),))
    report = fit(Xoshiro(2400), problem; sampling=MonteCarlo(128))
    validation = validate(Xoshiro(2401), report; plan=MonteCarlo(256))
    @test only(only(validation.checks)).status == :undefined
    auxiliary = Elicitation(
        (@vn(theta.global_scale) => (Approximate(:mean, Mean(), 1.0; scale=1.0),),),
        (:theta => (:law => HorseshoeFamily(2; scale=0.3),),),
    )
    auxiliary_report = fit(Xoshiro(2406), auxiliary; sampling=MonteCarlo(128))
    @test only(
        only(validate(Xoshiro(2407), auxiliary_report; plan=MonteCarlo(256)).checks),
    ).status == :undefined
    joint_problem = Elicitation(
        beliefs,
        (
            :theta => (:law => HorseshoeFamily(2; scale=0.3),),
            :noise => (:normal => NormalFamily(location=0.0, scale=1.0),),
        );
        combinations=(:joint => Independent(:theta => :law, :noise => :normal),),
    )
    joint_report = fit(Xoshiro(2402), joint_problem; sampling=MonteCarlo(128))
    @test only(only(validate(Xoshiro(2403), joint_report).checks)).status == :undefined
    repeated = Elicitation(
        (@vn(theta[1].coefficients[1]) => (Approximate(:mean, Mean(), 0.0; scale=1.0),),),
        (:theta => (:law => IIDFamily(HorseshoeFamily(2; scale=0.3), 2),),),
    )
    repeated_report = fit(Xoshiro(2408), repeated; sampling=MonteCarlo(128))
    @test only(
        only(validate(Xoshiro(2409), repeated_report; plan=MonteCarlo(256)).checks),
    ).status == :undefined

    variance = (@vn(theta.coefficients) => (Approximate(:std, Std(), ones(2); scale=1.0),),)
    cases = (
        (RegularizedHorseshoeFamily(2; scale=0.3, slab_scale=1.0, slab_df=0.8), :undefined),
        (RegularizedHorseshoeFamily(2; scale=0.3, slab_scale=1.0, slab_df=1.5), :target),
        (
            R2D2Family(
                2;
                noise_scale=1.0,
                predictor_scales=ones(2),
                mean_r2=0.5,
                precision_r2=2.0,
            ),
            :undefined,
        ),
        (GeneralizedDoubleParetoFamily(2; shape=2.0, scale=0.4), :undefined),
    )
    for (family, expected) in cases
        problem = Elicitation(variance, (:theta => (:law => family,),))
        report = fit(Xoshiro(2404), problem; sampling=MonteCarlo(256))
        @test only(only(validate(Xoshiro(2405), report).checks)).status == expected
    end
end
