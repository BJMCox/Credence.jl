function Credence._compound_base(
    f::Union{Credence.HorseshoeFamily,Credence.RegularizedHorseshoeFamily},
    p,
)
    scale = Credence._compound_parameter(f, p, Val(:scale))
    unit = one(scale)
    half_cauchy =
        asmeasure(truncated(Cauchy(zero(unit), unit), zero(unit), oftype(float(unit), Inf)))
    parts = (
        z=MeasureBase.productmeasure(
            fill(asmeasure(Normal(zero(unit), unit)), f.dimension),
        ),
        global_scale=half_cauchy,
        local_scales=MeasureBase.productmeasure(fill(half_cauchy, f.dimension)),
    )
    if f isa Credence.RegularizedHorseshoeFamily
        shape = f.slab_df / 2
        parts = (; parts..., slab_aux=asmeasure(Distributions.InverseGamma(shape, shape)))
    end
    MeasureBase.productmeasure(parts)
end

function Credence._compound_linear(f::Credence.HorseshoeFamily, p, x)
    scale = Credence._compound_parameter(f, p, Val(:scale)) * x.global_scale
    Diagonal(scale .* x.local_scales)
end

function _regularized_horseshoe_scale(log_scale, log_slab)
    if log_scale < log_slab
        return exp(log_scale - log1p(exp(2 * (log_scale - log_slab))) / 2)
    end
    exp(log_slab - log1p(exp(2 * (log_slab - log_scale))) / 2)
end
function Credence._compound_linear(f::Credence.RegularizedHorseshoeFamily, p, x)
    log_scale = log(Credence._compound_parameter(f, p, Val(:scale))) + log(x.global_scale)
    log_slab =
        log(Credence._compound_parameter(f, p, Val(:slab_scale))) + log(x.slab_aux) / 2
    # Combine on the log scale before multiplication can lose compensating factors.
    scales = map(x.local_scales) do local_scale
        _regularized_horseshoe_scale(log_scale + log(local_scale), log_slab)
    end
    Diagonal(scales)
end
