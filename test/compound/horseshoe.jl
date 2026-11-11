@testset "Horseshoe joint prior" begin
    family = HorseshoeFamily(2; scale=0.3)
    report = fit(Xoshiro(171), Elicitation((), (:theta => (:horseshoe => family,),)))
    prior = only(report.results).measure
    point = (coefficients=[0.2, -0.5], global_scale=0.7, local_scales=[0.4, 1.8])
    half_cauchy = truncated(Cauchy(), 0.0, Inf)
    scales = 0.3 .* point.global_scale .* point.local_scales
    expected =
        sum(logpdf.(Normal.(0.0, scales), point.coefficients)) +
        logpdf(half_cauchy, point.global_scale) +
        sum(logpdf.(Ref(half_cauchy), point.local_scales))
    @test logdensityof(prior, point) ≈ expected

    rng = Xoshiro(172)
    draw = rand(rng, prior)
    @test all(isfinite, draw.coefficients)
    @test draw.global_scale > 0 && all(>(0), draw.local_scales)

    fitted = HorseshoeFamily(2; starts=(log_scale=log(0.3),))
    fitted_prior =
        only(fit(Xoshiro(173), Elicitation((), (:theta => (:law => fitted,),))).results).measure
    @test logdensityof(fitted_prior, point) ≈ expected
end

@testset "Regularized horseshoe joint prior" begin
    family = RegularizedHorseshoeFamily(2; scale=0.3, slab_scale=1.2, slab_df=4.0)
    report = fit(Xoshiro(174), Elicitation((), (:theta => (:regularized => family,),)))
    prior = only(report.results).measure
    point =
        (coefficients=[0.2, -0.5], global_scale=0.7, local_scales=[0.4, 1.8], slab_aux=1.4)
    half_cauchy = truncated(Cauchy(), 0.0, Inf)
    raw = 0.3 .* point.global_scale .* point.local_scales
    slab_variance = 1.2^2 * point.slab_aux
    scales = sqrt.(slab_variance .* raw .^ 2 ./ (slab_variance .+ raw .^ 2))
    expected =
        sum(logpdf.(Normal.(0.0, scales), point.coefficients)) +
        logpdf(half_cauchy, point.global_scale) +
        sum(logpdf.(Ref(half_cauchy), point.local_scales)) +
        logpdf(InverseGamma(2.0, 2.0), point.slab_aux)
    @test logdensityof(prior, point) ≈ expected

    fitted =
        RegularizedHorseshoeFamily(2; starts=(log_scale=log(0.3), log_slab_scale=log(1.2)))
    fitted_prior =
        only(fit(Xoshiro(175), Elicitation((), (:theta => (:law => fitted,),))).results).measure
    @test logdensityof(fitted_prior, point) ≈ expected

    # For very large local scales the conditional law tends to the finite slab.
    extreme = merge(point, (local_scales=[1.0e150, 1.0e150],))
    slab = Normal(0.0, sqrt(slab_variance))
    expected_extreme =
        sum(logpdf.(Ref(slab), point.coefficients)) +
        logpdf(half_cauchy, point.global_scale) +
        sum(logpdf.(Ref(half_cauchy), extreme.local_scales)) +
        logpdf(InverseGamma(2.0, 2.0), point.slab_aux)
    @test logdensityof(prior, extreme) ≈ expected_extreme

    large_family = RegularizedHorseshoeFamily(1; scale=1.0e200, slab_scale=1.0e200)
    large_prior = only(
        fit(Xoshiro(177), Elicitation((), (:theta => (:law => large_family,),))).results,
    ).measure
    auxiliary = (global_scale=1.0e200, local_scales=[1.0e-200], slab_aux=1.0)
    conditional = MeasureBase.kernelmeasure(MeasureBase.bindkernel(large_prior), auxiliary)
    expected_scale = setprecision(BigFloat, 256) do
        raw = big"1.0e200" * big"1.0e200" * big"1.0e-200"
        slab = big"1.0e200"
        Float64(inv(sqrt(inv(raw^2) + inv(slab^2))))
    end
    @test logdensityof(conditional, (coefficients=[1.0e200],)) ≈
          logpdf(Normal(0.0, expected_scale), 1.0e200)

    unit_family = RegularizedHorseshoeFamily(1; scale=1.0, slab_scale=1.0)
    unit_prior = only(
        fit(Xoshiro(178), Elicitation((), (:theta => (:law => unit_family,),))).results,
    ).measure
    function conditional_density(scales)
        auxiliary = (global_scale=scales[1], local_scales=[1.0], slab_aux=scales[2]^2)
        conditional =
            MeasureBase.kernelmeasure(MeasureBase.bindkernel(unit_prior), auxiliary)
        logdensityof(conditional, (coefficients=[0.0],))
    end
    @test ForwardDiff.hessian(conditional_density, [1.0, 1.0]) ≈ [1.0 -0.5; -0.5 1.0]
end
