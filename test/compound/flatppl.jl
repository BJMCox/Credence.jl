@testset "Compound FlatPPL expression" begin
    family = GeneralizedDoubleParetoFamily(2; shape=3.0, scale=0.4)
    prior =
        only(fit(Xoshiro(1909), Elicitation((), (:theta => (:gdp => family,),))).results).measure
    expected = """
    jointchain(
        joint(
            local_rates = iid(Gamma(3.0, 1.0), 2)
        ),
        r -> joint(coefficients = joint(
            Laplace(0.0, 1.2000000000000002 / r.local_rates[1]),
            Laplace(0.0, 1.2000000000000002 / r.local_rates[2])
        ))
    )"""
    @test flatppl(prior) == expected
end
