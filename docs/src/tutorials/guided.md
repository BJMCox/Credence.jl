# [Guided elicitation at the Julia REPL](@id guided-tutorial)

This session follows one expert from questions to feedback, a revised answer, and an explicit
choice. It uses the same API and final answers as the [programmatic workflow](@ref scalar-tutorial).
Run the commands in a Julia REPL. The outputs below come from executing the session.

From a Credence checkout, install the example environment and start Julia:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples
```

```@repl guided
using Credence, MeasureBase, Optimization, OptimizationOptimJL, Random
using Distributions: Normal, LocationScale, TDist
include(joinpath(pkgdir(Credence), "examples", "feedback.jl"));
```

## Agree on the quantity

Ask: **What are we uncertain about, and under which conditions?**

Here the quantity is a sensor's fixed mean offset from a reference at 20°C, in degrees Celsius.
Zero means no offset. Positive values mean the sensor reads higher on average.
The prior describes uncertainty about this fixed offset. It does not describe the variation
of individual readings around that mean.

Before collecting answers, check the probability interpretation together. A lower quartile
places 25% of prior probability at or below that value. It is neither the most likely value
nor a lower limit on possible offsets.

## Ask about quartiles

Ask: **Below which offset would you place 25% of your probability?**
The illustrative answer is **-0.5°C**.

Ask: **Below which offset would you place 75% of your probability?**
The initial answer is **1.5°C**.

Pause if the expert cannot judge either question. Record it as unknown and discuss the
context. This two-quartile example needs both answers before fitting. A missing answer
does not mean zero, an interval midpoint, or a uniform prior.

```@repl guided
initial_answers = (q25=-0.5, q75=1.5)
```

Read the interpretation back: **You assign 25% probability to an offset at or below -0.5°C,
and 75% to an offset at or below 1.5°C. Is that what you mean?**

Turn confirmed answers into beliefs. Equal importance is an explicit analyst choice here.
The discrepancy scale is one degree Celsius. These are approximate targets, not hard bounds.

```@repl guided
function beliefs_for(answers)
    return (
        Approximate(
            :lower_quartile,
            Quantile(0.25),
            answers.q25;
            scale=1.0,
            importance=1.0,
        ),
        Approximate(
            :upper_quartile,
            Quantile(0.75),
            answers.q75;
            scale=1.0,
            importance=1.0,
        ),
    )
end;
```

## Present alternative families

Compare a Normal family and a Student-t family. Five degrees of freedom is a family
assumption agreed for this demonstration. Two quartiles do not determine tail behavior.
Both families allow positive and negative offsets. A log scale keeps proposed scales positive.

```@repl guided
candidates = (
    Candidate(
        :normal,
        p -> AbstractMeasure(Normal(p[1], exp(p[2]))),
        ([0.0, 0.0], [1.0, -0.5]),
    ),
    Candidate(
        :student_t,
        p -> AbstractMeasure(LocationScale(p[1], exp(p[2]), TDist(5))),
        ([0.0, 0.0], [1.0, -0.5]),
    ),
);
initial_problem = Elicitation(beliefs_for(initial_answers), candidates);
initial_report = fit(Xoshiro(42), initial_problem, NelderMead(); maxiters=2000, g_tol=1e-12);
[(result.id, result.error) for result in initial_report.results]
```

Each start retains its own result or error. Inspect errors before accessing a result's measure.
These starts succeed. Their solver return codes remain available as `result.solver.retcode`.

## Discuss feedback beyond the questions

Ask: **Do these curves represent your uncertainty? What probability do they assign to an
offset above 4°C, which we have not asked about?**

```@repl guided
[
    (result.id, round(evaluate(Probability(4, Inf), result.measure).value; digits=4)) for
    result in initial_report.results
]
initial_figure = feedback(initial_report);
mkpath("elicitation-output");
save("elicitation-output/initial-prior.svg", initial_figure);
```

![Initial prior density and cumulative probability for each retained fit.](elicitation-output/initial-prior.svg)

The quartiles fit closely in both families, while their tails differ. The dashed lines in
the probability plot mark 25% and 75%. Every curve is one retained fit, not an envelope,
credible band, or mixture of candidate families. The displayed x-range is not a support bound.
The SVG file in `elicitation-output` also lets you review the plot outside Julia.

Keep feedback separate from judgments. Showing a tail probability does not create a new
belief. Ask whether the expert wants to affirm, revise, or leave an answer unresolved.
If the expert supplies a new hard probability requirement, record it explicitly and use
a solver that supports constraints, as described in [Beliefs and validation](@ref validation-guide).

## Revise an answer without losing the first fit

Suppose the expert reconsiders the upper quartile using their calibration experience and
changes it to **1.0°C**. Record the reason. This is a revised judgment, not a new observation.

```@repl guided
revision_reason = "After reviewing large offsets, the expert revised the upper quartile from calibration experience.";
revised_answers = merge(initial_answers, (q75=1.0,))
revised_problem = Elicitation(beliefs_for(revised_answers), candidates);
revised_report = fit(Xoshiro(42), revised_problem, NelderMead(); maxiters=2000, g_tol=1e-12);
[(result.id, result.error) for result in revised_report.results]
```

The original answers and report remain available. Review the new feedback rather than
carrying the earlier assessment forward automatically:

```@repl guided
(original=initial_answers, revised=revised_answers)
revised_figure = feedback(revised_report);
save("elicitation-output/revised-prior.svg", revised_figure);
comparison = compare(revised_report);
comparison.incomparable
```

![Revised prior density and cumulative probability for each retained fit.](elicitation-output/revised-prior.svg)

`compare` preserves tradeoffs for each target. Small solver residual differences do not
identify the best prior family. The expert still needs to assess the assumptions and tails.

## Validate the current report

Validation belongs to a specific report. A check of the initial report is not a check of the revision:

```@repl guided
initial_validation = validate(Xoshiro(43), initial_report);
initial_validation.report === revised_report
validation = validate(Xoshiro(43), revised_report);
[[check.status for check in row] for row in validation.checks]
```

These checks are `:target`: approximate targets do not provide pass/fail tolerances.
Inspect the achieved values and solver return codes before choosing:

```@repl guided
[
    (
        id=result.id,
        quartiles=map(
            p -> round(evaluate(Quantile(p), result.measure).value; digits=3),
            (0.25, 0.75),
        ),
        upper_tail=round(evaluate(Probability(4, Inf), result.measure).value; digits=4),
        solver=string(result.solver.retcode),
    ) for result in revised_report.results
]
```

Rounding changes only the display. Use `Hard` beliefs when a requirement must hold.
An accepted selection requires this report's validation and satisfaction of all hard requirements.

## Make a choice and use the prior

Ask: **Which retained prior do you accept, and why?**

For this demonstration, the expert accepts the first Normal fit after reviewing both families.
The ID and rationale are supplied decisions. Neither comes from a ranking rule.

```@repl guided
selection = select(
    revised_report,
    (:normal, 1);
    validation,
    rationale="The expert accepts the Normal family after reviewing both families and their tails.",
);
prior = measure(selection);
rand(Xoshiro(44), prior^5)
```

Use `role=:reference` instead when retaining a candidate for discussion without accepting it.
If a judgment changes again, create a new problem, review its feedback, validate its report,
and make a new selection. Keep the old records under their original answers.

The programmatic example starts from these final answers, uses the same starts and fitting
settings, and records this same decision. Keep the questions, answers, revision reason,
script, seeds, and pinned environment together for replay. Portable report files and a
browser-based elicitation interface remain future work.

The selected object is a native MeasureBase measure. The [Wren tutorial](@ref wren-tutorial)
shows how to use a selected joint measure in a model.
