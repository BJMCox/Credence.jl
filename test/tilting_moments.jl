@testset "Event tilt moment validation" begin
    for (upper, ratio, expected) in ((Inf, 2.0, :undefined), (1.0, 4.0, :target))
        report = tilt(
            asmeasure(Cauchy()),
            (Hard(:event, Probability(0, upper), EqualTo(1.0)),),
            BoundedKL(lower=0, upper=ratio),
        )
        problem = Elicitation(
            (Approximate(:mean, Mean(), 2log(2)/pi; scale=1.0),),
            (Candidate(:tilted, _ -> measure(report), [0.0]),),
        )
        validation =
            validate(Xoshiro(15), fit(Xoshiro(16), problem); sampling=MonteCarlo(1024))
        check = only(only(validation.checks))
        @test check.status == expected
        if expected == :target
            variance = 4/pi - 1 - (2log(2)/pi)^2
            @test abs(check.estimate.value - 2log(2)/pi) < 6sqrt(variance/1024)
        end
    end
end
