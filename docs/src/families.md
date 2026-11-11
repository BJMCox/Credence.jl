# [Candidate families](@id family-catalogue)

Family descriptors define the measures considered during elicitation and their fitted
parameters. Load `Distributions` to use the built-in adapters. Each family accepts `starts`
for initial parameter values. See [Named inputs](@ref input-guide) for associating families
with parameter references and beliefs.

## Families by parameter type

The table suggests candidate families to compare. Elicited beliefs determine their
parameters, while the scientific model determines support and any physical bounds.

| Example parameter                             | Recommended candidate families                                                                                           | Choice to examine                            |
|:--------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------ |:-------------------------------------------- |
| Event rate or positive intensity              | [`GammaFamily`](@ref), [`LogNormalFamily`](@ref)                                                                         | Mass near zero and the upper tail            |
| Probability, efficiency, or proportion        | [`ScaledBetaFamily(0.0, 1.0)`](@ref ScaledBetaFamily)                                                                    | Concentration near either boundary           |
| Time to an event or a characteristic lifetime | [`GammaFamily`](@ref), [`WeibullFamily`](@ref), [`LogNormalFamily`](@ref)                                                | Skewness and plausible long times            |
| Noise standard deviation or group-level scale | [`HalfNormalFamily`](@ref), [`HalfStudentTFamily`](@ref)                                                                 | Allowance for large scales                   |
| Signed effect, offset, or location            | [`NormalFamily`](@ref), [`StudentTFamily`](@ref)                                                                         | Allowance for large deviations               |
| Location or amplitude with physical bounds    | [`ScaledBetaFamily(lower, upper)`](@ref ScaledBetaFamily), [`TruncatedFamily(base, lower, upper)`](@ref TruncatedFamily) | Shape within the known bounds                |
| Mixture fractions summing to one              | [`DirichletFamily`](@ref)                                                                                                | Mean fractions and their joint concentration |
| Correlated signed effects                     | [`MvNormalFamily`](@ref), [`MvStudentTFamily`](@ref)                                                                     | Dependence and joint tail behaviour          |

Set scales in the parameter's physical units and compare the implied prior predictions.
Student-t families provide heavier tails than their Normal counterparts. Physical bounds
describe possible values; quantile beliefs describe uncertainty within that support.
The [prior choice guidance](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations)
discusses scale, tails, and the distinction between physical bounds and prior uncertainty.

## Scalar families

The table lists each family's support and parameter keywords. A numeric keyword fixes that
parameter. A keyword set to `nothing`, the default, leaves the parameter free for fitting.

| Family                                                    | Support                          | Parameter keywords                   |
|:--------------------------------------------------------- |:-------------------------------- |:------------------------------------ |
| [`NormalFamily`](@ref)                                    | Real line                        | `location`, `scale`                  |
| [`LogNormalFamily`](@ref)                                 | Positive real line               | `location`, `scale` on the log scale |
| [`GammaFamily`](@ref)                                     | Nonnegative real line            | `shape`, `scale`                     |
| [`WeibullFamily`](@ref)                                   | Nonnegative real line            | `shape`, `scale`                     |
| [`ExponentialFamily`](@ref)                               | Nonnegative real line            | `scale`                              |
| [`StudentTFamily`](@ref)                                  | Real line                        | `df`, `location`, `scale`            |
| [`CauchyFamily`](@ref)                                    | Real line                        | `location`, `scale`                  |
| [`LaplaceFamily`](@ref)                                   | Real line                        | `location`, `scale`                  |
| [`HalfNormalFamily`](@ref)                                | Nonnegative real line            | `scale`                              |
| [`HalfStudentTFamily`](@ref)                              | Nonnegative real line            | `df`, `scale`                        |
| [`ScaledBetaFamily(lower, upper)`](@ref ScaledBetaFamily) | Interval from `lower` to `upper` | Fitted Beta shapes; fixed bounds     |

Normal `scale` is the standard deviation. LogNormal `location` and `scale` are the mean
and standard deviation of the logarithm. Gamma uses shape and scale, with mean
`shape * scale`. Exponential `scale` is its mean. Weibull uses shape and scale, with
cumulative probability `1 - exp(-(x / scale)^shape)` for nonnegative `x`. Laplace `scale`
gives standard deviation `sqrt(2) * scale`.

Student-t uses degrees of freedom `df` and a location-scale transformation of the standard
Student-t law. Its mean exists for `df > 1`, and its variance is
`scale^2 * df / (df - 2)` for `df > 2`. Cauchy quantiles remain defined despite its
undefined mean and variance. Half-Normal and half-Student-t are the corresponding
zero-location laws, normalized over nonnegative values. Their `scale` is the parent law's
scale. Quantiles and event probabilities can express beliefs when moments are undefined.

`ScaledBetaFamily(lower, upper)` maps a Beta law on the unit interval to the specified
interval. Its two shape parameters are fitted. Use `ScaledBetaFamily(0.0, 1.0)` for
probabilities and proportions.

## Vector families

The dimension `n` is fixed. The table distinguishes fitted parameters from optional fixed
parameter keywords.

| Family                                           | Support                                       | Fitted parameters and fixed options                    |
|:------------------------------------------------ |:--------------------------------------------- |:------------------------------------------------------ |
| [`MvNormalFamily(n)`](@ref MvNormalFamily)       | Real vectors of length `n`                    | Mean and full covariance                               |
| [`MvLogNormalFamily(n)`](@ref MvLogNormalFamily) | Positive vectors of length `n`                | Mean and full covariance of the logarithms             |
| [`MvStudentTFamily(n)`](@ref MvStudentTFamily)   | Real vectors of length `n`                    | Location, scatter, and `df`; optional fixed `df`       |
| [`DirichletFamily(n)`](@ref DirichletFamily)     | Positive vectors of length `n` summing to one | Concentrations; optional fixed `mean`, `concentration` |

Multivariate Student-t uses a scatter matrix. Its covariance is `df / (df - 2)` times
that matrix for `df > 2`. Its mean exists for `df > 1`. Multivariate LogNormal covariance
parameters describe the logarithms rather than the original variables.

`DirichletFamily(n)` fits all concentration parameters. Fixing `mean` fits only the total
concentration. Fixing `concentration` fits the mean on the simplex. Supplying both defines
a fixed law. The mean must be a positive vector summing to one, and `concentration` must
be positive.

Indexed component beliefs use analytic marginal laws: Normal, LogNormal, Student-t, and
Beta for the four vector families above. Vector subsets preserve their original component
values and dependence. In particular, a Dirichlet subset retains its original scale and
negative covariances. Vector quantile beliefs refer to componentwise marginal quantiles.

## Truncation and independent arrays

| Constructor                                                   | Result                                                        |
|:------------------------------------------------------------- |:------------------------------------------------------------- |
| [`TruncatedFamily(base, lower, upper)`](@ref TruncatedFamily) | Fits a scalar base family, normalized within the fixed bounds |
| [`IIDFamily(family, dimensions)`](@ref IIDFamily)             | Fits one family shared by independent array entries           |

For truncation, `base` accepts a scalar family descriptor or a supported Distributions
constructor, such as `Distributions.LogNormal`. Beliefs apply to the resulting truncated
law. For an iid array, pass an integer for a vector length or a tuple for its shape.
Separate parameter references fit separate laws when entries need different parameters.

```@example family-catalogue
using Credence
import Distributions

families = (
    GammaFamily(shape=2.0),
    StudentTFamily(df=4.0, location=0.0),
    HalfStudentTFamily(df=3.0),
    DirichletFamily(3; mean=[0.2, 0.3, 0.5]),
    MvStudentTFamily(2; df=4.0),
    TruncatedFamily(Distributions.LogNormal, 0.0, 8.0),
    IIDFamily(LogNormalFamily(), (2, 3)),
)
nothing # hide
```

## Starting coordinates

Supply a named record for one start or a tuple of records for several starts. Scalar
families use `location` directly and log coordinates for each free positive parameter:
`log_scale`, `log_shape`, or `log_df`. A fixed parameter has no starting coordinate.
For `ScaledBetaFamily`, use `log_alpha` and `log_beta` for the Beta shapes.

Vector Normal, LogNormal, and Student-t families use `location`, `log_scales`, and `lower`.
These describe a lower triangular factor `L`: `exp.(log_scales)` forms its diagonal,
and `lower` fills its strict lower triangle column by column. The covariance is `L * L'`
for Normal, the log covariance for LogNormal, and the scatter matrix for Student-t.
A fitted Student-t `df` adds `log_df`.

Dirichlet coordinates depend on its fixed parameters:

| Fixed parameters | Starting coordinates                                          |
|:---------------- |:------------------------------------------------------------- |
| Neither          | `log_alpha`, a vector of length `n`                           |
| `mean`           | Scalar `log_concentration`                                    |
| `concentration`  | `logits`, a vector of length `n - 1`; the final logit is zero |
| Both             | An empty named record                                         |

```@example family-catalogue
gamma = GammaFamily(shape=2.0, starts=(log_scale=log(3.0),))
student = MvStudentTFamily(
    2;
    df=4.0,
    starts=(location=[0.0, 0.0], log_scales=log.([2.0, 1.0]), lower=[0.4]),
)
nothing # hide
```

## FlatPPL export

Scalar families and Dirichlet support FlatPPL export. Gamma and Exponential scales convert
to FlatPPL rates. Normal, LogNormal, and supported truncation and iid wrappers preserve
their measures. Multivariate Normal and LogNormal exports preserve their full dependence.
Multivariate Student-t export requires a corresponding FlatPPL primitive.
See the [FlatPPL example](@ref flatppl-tutorial) for a complete model and density check.
