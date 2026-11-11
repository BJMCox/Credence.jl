using LinearAlgebra, VarNames
import TOML
using Optimization, OptimizationOptimJL

module SessionTypeCollision
struct Mean end
struct Normal end
end

_session_array_builder(p) =
    asmeasure(MvNormal([1.0, 2.0] .+ p.location, [1.0 0.2; 0.2 2.0]))
_session_total(weights) = sum(weights)
_session_normal_builder(location) =
    p -> asmeasure(Normal(location+p.location, exp(p.log_scale)))
_session_failure_builder(p) = error("Rejected candidate")
_session_other_builder(p) = asmeasure(Normal(p.location-1.0, exp(p.log_scale)))
const _session_global_shift=Ref(0.0)
_session_global_builder(p) = asmeasure(Normal(p.location+_session_global_shift[], 1.0))
const _session_same_line = (_ -> StdNormal(), _ -> StdUniform())

@testset "Portable elicitation session roundtrip" begin
    beliefs = (
        :weights => (
            Approximate(:mean, Mean(), Float32[1, 2]; scale=1.0f0),
            Hard(:bounds, Mean(), Between([nothing, 1.0], [2.0, missing])),
        ),
        Joint(@vn(weights[1]), @vn(weights[2])) => (
            Approximate(
                :covariance,
                CovarianceMatrix(),
                UpperTriangular(Float32[1 0.2; 0 2]);
                scale=1.0f0,
            ),
        ),
        Derived(:total, :weights, _session_total) =>
            (Approximate(:mean, Mean(), 3.0; scale=1.0),),
    )
    problem=Elicitation(
        beliefs,
        (:weights=>(Candidate(:joint, _session_array_builder, (location=0.0,)),),),
    )
    revised=replace(
        problem,
        :weights=>Approximate(:mean, Mean(), Float32[1.1, 2.1]; scale=1.0f0),
    )
    report=fit(Xoshiro(80), problem)
    validation=validate(Xoshiro(81), report; sampling=MonteCarlo(512))
    choice=select(
        report,
        :weights,
        (:joint, 1);
        validation,
        rationale="Reviewed the hierarchy",
    )
    callbacks=(array_builder=_session_array_builder, total=_session_total)
    metadata=(
        thresholds=Union{Missing,Float32}[missing, 1],
        shape=LowerTriangular([1.0 0; 0.2 2]),
        ratio=1//3,
    )
    mktempdir() do directory
        path=joinpath(directory, "elicitation.toml")
        save_session(
            path,
            problem;
            report,
            validation,
            selections=(choice,),
            callbacks,
            metadata,
            history=[
                (
                    problem=problem,
                    report=report,
                    validation=validation,
                    selections=(choice,),
                    rationale="Initial",
                ),
                (problem=revised, rationale="Revision"),
            ],
        )
        loaded=load_session(path; callbacks)
        @test loaded.report.problem === loaded.problem &&
              loaded.validation.report === loaded.report
        @test only(loaded.selections).report === loaded.report &&
              only(loaded.selections).validation === loaded.validation
        @test only(loaded.selections).rationale == choice.rationale
        @test logdensityof(measure(only(loaded.selections)), [0.5, 1.7]) ≈
              logdensityof(measure(choice), [0.5, 1.7])
        @test compare(loaded.report; validation=loaded.validation) ==
              compare(report; validation)
        @test loaded.validation.seed == validation.seed
        @test loaded.problem[:weights, :mean].target == Float32[1, 2]
        @test loaded.history[1].validation.report === loaded.history[1].report
        @test loaded.history[2].problem[:weights, :mean].target == Float32[1.1, 2.1]
        @test isequal(loaded.metadata, metadata)
        @test unlock(only(loaded.selections)).starts == unlock(choice).starts
    end
end

@testset "Callback recipes invalidate changed code or captures" begin
    builder=_session_normal_builder(0.0)
    problem=Elicitation(
        (Approximate(:mean, Mean(), 2.0; scale=1.0),),
        (
            Candidate(:normal, builder, (location=0.0, log_scale=0.0)),
            Candidate(:failure, _session_failure_builder, [0.0]),
        ),
    )
    report=fit(Xoshiro(82), problem, NelderMead(); maxiters=1000)
    validation=validate(Xoshiro(83), report)
    callbacks=(normal=(callback=builder, version="1"), failure=_session_failure_builder)
    mktempdir() do directory
        path=joinpath(directory, "fit.toml")
        save_session(path, problem; report, validation, callbacks)
        loaded=load_session(path; callbacks)
        @test loaded.report.results[1].solver.retcode ==
              string(report.results[1].solver.retcode)
        @test mean(loaded.report.results[1].measure) ≈ mean(report.results[1].measure)
        changed=(
            normal=(callback=_session_normal_builder(1.0), version="1"),
            failure=_session_failure_builder,
        )
        @test_throws ArgumentError load_session(path; callbacks=changed)
        changed_code=(
            normal=(callback=_session_other_builder, version="1"),
            failure=_session_failure_builder,
        )
        @test_throws ArgumentError load_session(path; callbacks=changed_code)
        @test occursin(
            "Rejected candidate",
            sprint(showerror, loaded.report.results[2].error),
        )
        save_session(path, loaded; callbacks)
        repeated=load_session(path; callbacks)
        @test repeated.report.results[2].error.type == loaded.report.results[2].error.type
    end
end

@testset "Callbacks on one source line retain separate identities" begin
    original, changed = _session_same_line
    problem = Elicitation(
        (Hard(:mean, Mean(), Between(-0.1, 0.1)),),
        (Candidate(:normal, original, [0.0]),),
    )
    report = fit(Xoshiro(98), problem)
    validation = validate(Xoshiro(99), report)
    selection = select(report, (:normal, 1); validation, rationale="Centered prior")
    mktempdir() do directory
        path = joinpath(directory, "callback.toml")
        save_session(
            path,
            problem;
            report,
            validation,
            selections=(selection,),
            callbacks=(law=original,),
        )
        @test mean(
            measure(only(load_session(path; callbacks=(law=original,)).selections)),
        ) == 0
        @test_throws ArgumentError load_session(path; callbacks=(law=changed,))
    end
end

@testset "Saved linear ranges preserve their lengths and values" begin
    problem = Elicitation((), ())
    ranges = (LinRange(0.0, 1.0, 0), LinRange(2.0, 2.0, 1), LinRange(1.0, 0.0, 5))
    mktempdir() do directory
        path = joinpath(directory, "ranges.toml")
        save_session(path, problem; metadata=(ranges=ranges,))
        @test isequal(load_session(path).metadata.ranges, ranges)
    end
end

@testset "Standard family session remains usable" begin
    problem=Elicitation(
        (:x=>(Approximate(:median, Quantile(0.5), 0.0; scale=1.0),),),
        (:x=>(:normal=>NormalFamily(location=0.0),),),
    )
    report=fit(Xoshiro(84), problem)
    validation=validate(Xoshiro(85), report)
    selection=select(report, :x, (:normal, 1); validation, rationale="Centered prior")
    mktempdir() do directory
        path=joinpath(directory, "normal.toml")
        save_session(path, problem; report, validation, selections=(selection,))
        loaded=load_session(path)
        @test evaluate(Quantile(0.5), measure(only(loaded.selections))).value == 0
        @test compare(loaded.report; validation=loaded.validation) ==
              compare(report; validation)
        fresh=validate(Xoshiro(86), fit(Xoshiro(87), loaded.problem))
        @test all(row -> only(row).status==:target, fresh.checks)
        edited=TOML.parsefile(path)
        edited["problem"]["beliefs"]["values"][1]["target"]["value"]="2.0"
        open(path, "w") do io
            TOML.print(io, edited; sorted=true)
        end
        @test_throws ArgumentError load_session(path)
        @test_throws ArgumentError save_session(
            path,
            problem;
            metadata=(collision=SessionTypeCollision.Mean(),),
        )
        @test_throws ArgumentError save_session(
            path,
            problem;
            metadata=(collision=SessionTypeCollision.Normal,),
        )
    end
end

@testset "Composed family definitions survive saving" begin
    problem=Elicitation(
        (
            :rate=>(Approximate(:mean, Mean(), 4.0; scale=1.0),),
            :weights=>(Approximate(:mean, Mean(), [0.2, 0.3, 0.5]; scale=0.1),),
            :coefficients=>(Approximate(:mean, Mean(), [0.0, 0.0]; scale=1.0),),
        ),
        (
            :rate=>(:gamma=>GammaFamily(shape=2.0, scale=2.0),),
            :weights=>(
                :dirichlet=>DirichletFamily(3; mean=[0.2, 0.3, 0.5], concentration=10.0),
            ),
            :coefficients=>(:student=>MvStudentTFamily(2; df=4.0),),
        );
        combinations=(
            :joint=>Independent(
                :rate=>:gamma,
                :weights=>:dirichlet,
                :coefficients=>:student,
            ),
        ),
    )
    report=fit(Xoshiro(88), problem)
    validation=validate(Xoshiro(89), report)
    selection=select(
        report,
        (:joint, 1);
        validation,
        rationale="Independent calibrated factors",
    )
    mktempdir() do directory
        path=joinpath(directory, "joint.toml")
        save_session(path, problem; report, validation, selections=(selection,))
        loaded=load_session(path)
        point=(rate=3.0, weights=[0.25, 0.25, 0.5], coefficients=[0.2, -0.4])
        @test logdensityof(measure(only(loaded.selections)), point) ≈
              logdensityof(measure(selection), point)
        @test map(row -> row.value, belief_summary(loaded.validation)) ==
              map(row -> row.value, belief_summary(validation))
    end
end

_session_predictive(location) = asmeasure(Normal(location, 1.0))
@testset "Predictive sessions retain callbacks and can be saved again" begin
    prediction=Predictive(:observation, :location, _session_predictive)
    problem=Elicitation(
        (prediction=>(Approximate(:median, Quantile(0.5), 0.0; scale=1.0),),),
        (:location=>(:normal=>NormalFamily(location=0.0),),),
    )
    report=fit(Xoshiro(90), problem; sampling=MonteCarlo(128))
    validation=validate(Xoshiro(91), report)
    selection=select(
        report,
        :location,
        (:normal, 1);
        validation,
        rationale="Predictive median checked",
    )
    callbacks=(observation=_session_predictive,)
    mktempdir() do directory
        path=joinpath(directory, "predictive.toml")
        save_session(
            path,
            problem;
            report,
            validation,
            selections=(selection,),
            callbacks,
            metadata=Dict("assumption"=>"Normal observation noise"),
        )
        loaded=load_session(path; callbacks)
        save_session(path, loaded; callbacks)
        repeated=load_session(path; callbacks)
        @test only(repeated.selections).validation.report === repeated.report
        @test repeated.validation.seed == validation.seed
        @test repeated.validation.checks[1][1].estimate.value ==
              validation.checks[1][1].estimate.value
        @test repeated.metadata == Dict("assumption"=>"Normal observation noise")
        @test compare(repeated.report; validation=repeated.validation) ==
              compare(report; validation)
    end
end

@testset "Mutable global model data invalidates stored validation" begin
    problem=Elicitation(
        (Hard(:mean, Mean(), Between(-0.1, 0.1)),),
        (Candidate(:normal, _session_global_builder, (location=0.0,)),),
    )
    report=fit(Xoshiro(92), problem)
    validation=validate(Xoshiro(93), report)
    selection=select(report, (:normal, 1); validation, rationale="Centered model")
    callbacks=(normal=_session_global_builder,)
    mktempdir() do directory
        path=joinpath(directory, "model.toml")
        save_session(path, problem; report, validation, selections=(selection,), callbacks)
        _session_global_shift[]=10.0
        try
            @test_throws ArgumentError load_session(path; callbacks)
        finally
            _session_global_shift[]=0.0
        end
        @test mean(measure(only(load_session(path; callbacks).selections))) == 0.0
    end
end

_session_returned_value = 0.0
_session_constant_prediction(x) = _session_returned_value
@testset "Returned global observations keep session identity" begin
    prediction = Derived(:constant, :x, _session_constant_prediction)
    problem = Elicitation(
        (prediction => (Approximate(:mean, Mean(), 0.0; scale=1.0),),),
        (:x => (:normal => NormalFamily(location=0.0, scale=1.0),),),
    )
    report = fit(Xoshiro(96), problem; sampling=MonteCarlo(8))
    validation = validate(Xoshiro(97), report)
    callbacks = (constant=_session_constant_prediction,)
    mktempdir() do directory
        path = joinpath(directory, "global-observation.toml")
        save_session(path, problem; report, validation, callbacks)
        global _session_returned_value = 10.0
        try
            @test_throws ArgumentError load_session(path; callbacks)
        finally
            global _session_returned_value = 0.0
        end
        @test only(only(load_session(path; callbacks).validation.checks)).estimate.value ==
              0.0
    end
end
