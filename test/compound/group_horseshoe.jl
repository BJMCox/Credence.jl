@testset "Grouped global-local horseshoe" begin
    groups = (:detector, :detector, :detector, :calibration)
    family = GroupHorseshoeFamily(groups; scale=0.7)
    problem = Elicitation((), (:theta => (:grouped => family,),))
    prior = only(fit(Xoshiro(1220), problem).results).measure
    point = (
        coefficients=[0.3, -0.2, 0.5, 0.1],
        global_scale=0.8,
        group_scales=[0.6, 1.4],
        local_scales=[0.9, 1.1, 0.7, 1.2],
    )
    deviations = 0.7 .* 0.8 .* [0.6, 0.6, 0.6, 1.4] .* point.local_scales
    scales = [point.global_scale; point.group_scales; point.local_scales]
    expected =
        sum(logpdf.(Normal.(0.0, deviations), point.coefficients)) +
        sum(log(2) + logpdf(Cauchy(), x) for x in scales)
    @test logdensityof(prior, point) ≈ expected

    renamed = GroupHorseshoeFamily((12, 12, 12, 4); scale=0.7)
    renamed_prior = only(
        fit(Xoshiro(1221), Elicitation((), (:theta => (:grouped => renamed,),))).results,
    ).measure
    @test logdensityof(renamed_prior, point) ≈ expected

    free = GroupHorseshoeFamily(groups; starts=(log_scale=log(0.7),))
    free_prior =
        only(fit(Xoshiro(1222), Elicitation((), (:theta => (:grouped => free,),))).results).measure
    @test logdensityof(free_prior, point) ≈ expected

    wider = GroupHorseshoeFamily(groups; scale=1.4)
    wider_prior =
        only(fit(Xoshiro(1223), Elicitation((), (:theta => (:grouped => wider,),))).results).measure
    draw = rand(Xoshiro(1224), prior)
    wide_draw = rand(Xoshiro(1224), wider_prior)
    @test draw.coefficients .* 2 ≈ wide_draw.coefficients
    @test draw.group_scales == wide_draw.group_scales &&
          draw.local_scales == wide_draw.local_scales
end
