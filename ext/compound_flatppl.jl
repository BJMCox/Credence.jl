function Credence._flatppl(prior::Distributions.InverseGamma, path)
    shape, scale = Distributions.params(prior)
    "InverseGamma(" *
    Credence._flatppl_value(shape, path) *
    ", " *
    Credence._flatppl_value(scale, path) *
    ")"
end

_conditional_product(kind, scales) = Credence._flatppl_block(
    "joint(",
    (kind * "(0.0, " * scale * ")" for scale in scales),
    ")",
)

function Credence._flatppl_coefficients(f::Credence.HorseshoeFamily, p, r, path)
    scale = Credence._flatppl_value(Credence._compound_parameter(f, p, Val(:scale)), path)
    _conditional_product(
        "Normal",
        ("$scale * $r.global_scale * $r.local_scales[$i]" for i in 1:f.dimension),
    )
end
function Credence._flatppl_coefficients(f::Credence.RegularizedHorseshoeFamily, p, r, path)
    scale = Credence._flatppl_value(Credence._compound_parameter(f, p, Val(:scale)), path)
    slab =
        Credence._flatppl_value(Credence._compound_parameter(f, p, Val(:slab_scale)), path)
    scales = map(1:f.dimension) do i
        raw = "(log($scale) + log($r.global_scale) + log($r.local_scales[$i]))"
        cap = "(log($slab) + log($r.slab_aux) / 2.0)"
        "exp($raw + $cap - logsumexp([2.0 * $raw, 2.0 * $cap]) / 2.0)"
    end
    _conditional_product("Normal", scales)
end
function Credence._flatppl_coefficients(f::Credence.GroupHorseshoeFamily, p, r, path)
    scale = Credence._flatppl_value(Credence._compound_parameter(f, p, Val(:scale)), path)
    _conditional_product(
        "Normal",
        (
            "$scale * $r.global_scale * $r.group_scales[$group] * $r.local_scales[$i]" for
            (i, group) in enumerate(f.indices)
        ),
    )
end
function Credence._flatppl_coefficients(f::Credence.DirichletLaplaceFamily, p, r, path)
    scale = Credence._flatppl_value(Credence._compound_parameter(f, p, Val(:scale)), path)
    _conditional_product(
        "Laplace",
        ("$scale * $r.global_scale * $r.allocation[$i]" for i in 1:f.dimension),
    )
end
function Credence._flatppl_coefficients(f::Credence.R2D2Family, p, r, path)
    noise = Credence._flatppl_value(f.noise_scale, path)
    scales = map(1:f.dimension) do i
        predictor = Credence._flatppl_value(f.predictor_scales[i], path)
        "$noise * sqrt($r.allocation[$i] * $r.r2 / (2.0 * (1.0 - $r.r2))) / $predictor"
    end
    _conditional_product("Laplace", scales)
end
function Credence._flatppl_coefficients(f::Credence.NormalGammaFamily, p, r, path)
    _conditional_product("Normal", ("sqrt($r.local_variances[$i])" for i in 1:f.dimension))
end
function Credence._flatppl_coefficients(
    f::Credence.GeneralizedDoubleParetoFamily,
    p,
    r,
    path,
)
    shape = Credence._compound_parameter(f, p, Val(:shape))
    scale = Credence._compound_parameter(f, p, Val(:scale))
    multiplier = Credence._flatppl_value(shape * scale, path)
    _conditional_product(
        "Laplace",
        ("$multiplier / $r.local_rates[$i]" for i in 1:f.dimension),
    )
end
function Credence._flatppl_coefficients(f::Credence.MultiplicativeGammaFamily, p, r, path)
    scale = Credence._flatppl_value(Credence._compound_parameter(f, p, Val(:scale)), path)
    "Normal.(0.0, $scale .* exp.(-0.5 .* (log.($r.local_precisions) .+ addaxes(cumsum(log.($r.increments)), 1, 0))))"
end
function Credence._flatppl_coefficients(f::Credence.FunctionalHorseshoeFamily, p, r, path)
    scale = Credence._compound_parameter(f, p, Val(:scale))
    base = Credence._flatppl_value(f.baseline_scale^2, path)
    multiplier = Credence._flatppl_value(scale^2, path)
    n = size(f.design, 2)
    variances = [
        if i <= f.baseline_dimension
            base
        else
            "$multiplier * (1.0 - $r.shrinkage) / $r.shrinkage"
        end for i in 1:n
    ]
    loading = Credence._flatppl_value(f.loading, path)
    covariance = "$loading * diagmat([" * join(variances, ", ") * "]) * transpose($loading)"
    "MvNormal(zeros($n), $covariance)"
end
