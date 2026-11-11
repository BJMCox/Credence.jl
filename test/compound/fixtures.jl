function compound_cases()
    design = [1.0 -1.0; 1.0 0.0; 1.0 1.0]
    (
        HorseshoeFamily(2; scale=0.3),
        RegularizedHorseshoeFamily(2; scale=0.3, slab_scale=1.2),
        R2D2Family(
            2;
            noise_scale=0.5,
            predictor_scales=[1.0, 2.0],
            mean_r2=0.4,
            precision_r2=5.0,
        ),
        DirichletLaplaceFamily(2; scale=0.4),
        NormalGammaFamily(2; shape=0.5, scale=0.4),
        GroupHorseshoeFamily((:a, :b); scale=0.3),
        FunctionalHorseshoeFamily(design, design[:, 1]; baseline_scale=2.0, scale=0.4),
        MultiplicativeGammaFamily(2, 2; scale=0.3),
        GeneralizedDoubleParetoFamily(2; shape=3.0, scale=0.4),
    )
end
