function Credence._compound_base(family::Credence.MultiplicativeGammaFamily, parameters)
    first_shape = Credence._compound_parameter(family, parameters, Val(:first_shape))
    later_shape = Credence._compound_parameter(family, parameters, Val(:later_shape))
    local_df = Credence._compound_parameter(family, parameters, Val(:local_df))
    increments = map(1:family.columns) do column
        shape = if column == 1
            first_shape
        else
            later_shape
        end
        asmeasure(Gamma(shape, one(shape)))
    end
    MeasureBase.productmeasure((
        z=MeasureBase.productmeasure(
            fill(
                asmeasure(Normal(zero(local_df), one(local_df))),
                family.rows*family.columns,
            ),
        ),
        local_precisions=MeasureBase.productmeasure(
            fill(asmeasure(Gamma(local_df/2, 2/local_df)), family.rows, family.columns),
        ),
        increments=MeasureBase.productmeasure(increments),
    ))
end

function Credence._compound_linear(
    family::Credence.MultiplicativeGammaFamily,
    parameters,
    x,
)
    scale = Credence._compound_parameter(family, parameters, Val(:scale))
    log_precisions = log.(x.local_precisions) .+ transpose(cumsum(log.(x.increments)))
    Diagonal(vec(scale .* exp.(-log_precisions ./ 2)))
end
