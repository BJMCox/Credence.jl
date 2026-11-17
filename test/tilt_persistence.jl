using Optimization, OptimizationOptimJL
using IntervalSets: (..)

_saved_tilt_absent(x) = iszero(x.rate)
_saved_tilt_square(x) = x*x

@testset "Tilted-prior sessions" begin
    method=BoundedKL(lower=0.5, upper=2)
    beliefs=(Hard(:positive, Probability(0, Inf), Between(0.68, 0.72)),)
    report=tilt(StdNormal(), beliefs, method)
    validation=validate(Xoshiro(901), report)
    choice=select(report; validation, rationale="Reviewed calibration offset")
    @test choice.report===report && validation.report===report

    mktempdir() do directory
        path=joinpath(directory, "offset.toml")
        localvalidation=validate(Xoshiro(902), report; sampling=MonteCarlo(256))
        reference=select(
            report;
            role=:reference,
            validation=localvalidation,
            rationale="Retain the sampled check",
        )
        save_session(
            path,
            report;
            validation,
            selections=(choice, reference),
            history=(choice,),
        )
        loaded=load_session(path)
        selected, retained=loaded.selections
        @test selected.report===loaded.report && loaded.report.fit.problem===loaded.problem
        @test selected.validation===loaded.validation
        @test retained.validation.report===loaded.report &&
              retained.validation!==loaded.validation
        @test retained.validation.seed==localvalidation.seed
        @test selected.rationale==choice.rationale && retained.role==:reference
        @test measure(selected)===measure(loaded.report)
        @test loaded.report.normalization===report.normalization &&
              loaded.report.divergence===report.divergence &&
              loaded.report.ratio_bounds==report.ratio_bounds
        @test belief_summary(loaded.validation)==belief_summary(validation)
        @test flatppl(selected)==flatppl(choice)
        @test rand(Xoshiro(903), measure(selected)^64)==rand(
            Xoshiro(903),
            measure(choice)^64,
        )
        @test only(loaded.history).validation.report===only(loaded.history).report
        @test sprint(show, selected)==sprint(show, choice)
        rebuilt=fit(Xoshiro(904), Elicitation((), (unlock(selected),)))
        @test only(rebuilt.results).measure===measure(selected)
        save_session(path, loaded)
        repeated=load_session(path)
        @test repeated.report.normalization===report.normalization
        @test only(repeated.history).rationale==choice.rationale
    end

    @testset "Projection ownership and unresolved evidence" begin
        pending=TiltReport(
            report.reference,
            report.method,
            report.fit,
            report.normalization,
            report.divergence,
            report.ratio_bounds,
            report.feasibility,
            :unresolved,
        )
        checks=validate(Xoshiro(905), pending)
        @test_throws ArgumentError select(
            report;
            validation=checks,
            rationale="Different owner",
        )
        pending_choice=select(
            pending;
            role=:reference,
            validation=checks,
            rationale="Await convergence",
        )
        mktempdir() do directory
            path=joinpath(directory, "pending.toml")
            save_session(path, pending_choice)
            loaded=load_session(path)
            @test only(only(loaded.validation.checks)).status==:satisfied
            @test loaded.report.optimality==:unresolved
            @test_throws ArgumentError select(
                loaded.report;
                validation=loaded.validation,
                rationale="Still unresolved",
            )
        end
    end

    @testset "Finite atoms and fitted diagnostics" begin
        points=((rate=0.0, shape=1.0), (rate=1.0, shape=2.0), (rate=2.0, shape=3.0))
        reference=FiniteReference(points, Float32[1, 2, 3])
        finite=tilt(
            reference,
            (Hard(:absent, Probability(_saved_tilt_absent), EqualTo(0)),),
            BoundedKL(lower=0, upper=3),
            IPNewton(),
        )
        callbacks=(absent=_saved_tilt_absent,)
        mktempdir() do directory
            path=joinpath(directory, "atoms.toml")
            save_session(path, finite; callbacks)
            loaded=load_session(path; callbacks)
            @test loaded.report.reference.points==reference.points
            @test loaded.report.reference.weights==reference.weights
            @test logdensityof(measure(loaded.report), first(points))==-Inf
            rng, restored_rng=Xoshiro(906), Xoshiro(906)
            @test [rand(rng, measure(finite)) for _ in 1:64] == [rand(restored_rng, measure(loaded.report)) for _ in 1:64]
        end
        finite=tilt(
            FiniteReference((0.0, 1.0, 2.0), (1, 2, 1)),
            (Hard(:mean, Mean(), EqualTo(27/20)),),
            method,
            IPNewton();
            abstol=1e-12,
        )
        mktempdir() do directory
            path=joinpath(directory, "diagnostics.toml")
            save_session(path, finite)
            loaded=load_session(path)
            solver=only(loaded.report.fit.results).solver
            original=only(finite.fit.results).solver
            @test solver.evidence.dual_gap==original.dual_gap
            @test solver.evidence.residual==original.residual
            @test solver.evidence.residual_tolerance==original.residual_tolerance
            save_session(path, loaded)
            @test only(load_session(path).report.fit.results).solver.evidence==solver.evidence
        end
    end

    @testset "Native reference measures" begin
        for prior in (
            asmeasure(Normal(0.3, 0.8)),
            MeasureBase.normalize(truncate(StdNormal(), -1..1)),
        )
            report=tilt(prior, beliefs, method)
            mktempdir() do directory
                path=joinpath(directory, "reference.toml")
                save_session(path, report)
                loaded=load_session(path)
                @test logdensityof(measure(loaded.report), 0.4)==logdensityof(
                    measure(report),
                    0.4,
                )
                @test rand(Xoshiro(909), measure(loaded.report)^64)==rand(
                    Xoshiro(909),
                    measure(report)^64,
                )
            end
        end
    end

    @testset "Numerical normalization and callback versions" begin
        report=tilt(
            StdUniform(),
            (Hard(:second, Expectation(_saved_tilt_square), EqualTo(0.4)),),
            method,
            BFGS();
            g_tol=1e-10,
        )
        validation=validate(Xoshiro(907), report; sampling=MonteCarlo(512))
        choice=select(report; role=:reference, validation, rationale="Numerical projection")
        callbacks=(square=(callback=_saved_tilt_square, version="1"),)
        mktempdir() do directory
            path=joinpath(directory, "numerical.toml")
            save_session(path, choice; callbacks)
            loaded=load_session(path; callbacks)
            @test loaded.report.normalization===report.normalization
            @test loaded.report.normalization.evidence==:estimated
            @test loaded.report.feasibility==report.feasibility &&
                  loaded.report.optimality==report.optimality
            @test logdensity_estimate(loaded.report, 0.4)===logdensity_estimate(report, 0.4)
            @test_throws ArgumentError logdensityof(measure(loaded.report), 0.4)
            @test rand(Xoshiro(908), measure(loaded.report)^64)==rand(
                Xoshiro(908),
                measure(report)^64,
            )
            @test_throws ArgumentError select(
                loaded.report;
                validation=loaded.validation,
                rationale="Estimated normalization",
            )
            changed=(square=(callback=_saved_tilt_square, version="2"),)
            @test_throws ArgumentError load_session(path; callbacks=changed)
        end
    end
end
