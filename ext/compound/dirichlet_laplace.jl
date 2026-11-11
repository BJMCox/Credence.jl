function Credence._compound_base(family::Credence.DirichletLaplaceFamily, parameters)
    a = Credence._compound_parameter(family, parameters, Val(:concentration))
    MeasureBase.productmeasure((
        z=MeasureBase.productmeasure(
            fill(asmeasure(Laplace(zero(a), one(a))), family.dimension),
        ),
        global_scale=asmeasure(Gamma(family.dimension*a, 2one(a))),
        allocation=asmeasure(Dirichlet(fill(a, family.dimension))),
    ))
end

function Credence._compound_linear(family::Credence.DirichletLaplaceFamily, parameters, x)
    scale = Credence._compound_parameter(family, parameters, Val(:scale))
    Diagonal((scale*x.global_scale) .* x.allocation)
end
