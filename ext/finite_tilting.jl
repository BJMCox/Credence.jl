using LinearAlgebra

function _finite_entropy(q, p)
    sum(zip(q, p)) do (mass, reference)
        if mass < 0
            return oftype(mass, Inf)
        elseif iszero(mass)
            return zero(mass)
        end
        mass*log(mass/reference)
    end
end

_finite_zero_hessians!(h, q, _) = foreach(matrix -> fill!(matrix, 0), h)

function _finite_extreme(p, h, method, reverse)
    q = method.lower .* p
    remaining = one(eltype(q)) - sum(q)
    order = sortperm(h; rev=reverse)
    first = 1
    cutoff = h[order[first]]
    while first <= length(order) && remaining > 0
        last = first
        while last < length(order) && h[order[last+1]] == h[order[first]]
            last += 1
        end
        group = view(order, first:last)
        cutoff = h[order[first]]
        mass = sum(view(p, group))
        addition = min(remaining, (method.upper-method.lower)*mass)
        q[group] .+= (addition/mass) .* p[group]
        remaining -= addition
        first = last + 1
    end
    q, cutoff + dot(h .- cutoff, q)
end

function _finite_rows(points, p, beliefs, atol)
    rows, lower, upper = Vector{eltype(p)}[], eltype(p)[], eltype(p)[]
    for belief in beliefs
        feature = Credence._tilt_feature(belief.statistic)
        h = map(feature, points)
        all(x -> x isa Real && isfinite(x), h) || throw(
            ArgumentError("finite expectation features must return finite real values"),
        )
        lo, hi = Credence._bounds(belief.relation)
        center = first(h)
        scale = maximum(x -> abs(x-center), h)
        if iszero(scale)
            lo <= center <= hi || throw(
                ArgumentError(
                    "infeasible constant feature for belief $(belief.id): value $center",
                ),
            )
            continue
        end
        row = collect((h .- center) ./ scale)
        lo, hi = (lo-center)/scale, (hi-center)/scale
        if row[argmax(abs.(row))] < 0
            row, lo, hi = -row, -hi, -lo
        end
        duplicate = findfirst(r -> maximum(abs.(r-row)) <= 16eps(eltype(p)), rows)
        if duplicate === nothing
            push!(rows, row)
            push!(lower, lo)
            push!(upper, hi)
        else
            # Scaling can move the same floating-point equality by a few ulps.
            lower[duplicate] = max(lower[duplicate], lo)
            upper[duplicate] = min(upper[duplicate], hi)
            if lower[duplicate] > upper[duplicate]
                gap = lower[duplicate]-upper[duplicate]
                gap <= max(atol, maximum(abs.(rows[duplicate]-row))) || throw(
                    ArgumentError(
                        "infeasible parallel expectation bounds for belief $(belief.id)",
                    ),
                )
                lower[duplicate] = upper[duplicate] = (lower[duplicate]+upper[duplicate])/2
            end
        end
    end
    H = isempty(rows) ? zeros(eltype(p), 0, length(p)) : reduce(vcat, permutedims.(rows))
    H, lower, upper
end

function _finite_start(p, method, A, lower, upper, algorithm, atol; kwargs...)
    equalities = findall(lower .== upper)
    initial = p + A[equalities, :] \ (lower[equalities]-A[equalities, :]*p)
    margin = map(eachindex(lower)) do i
        if lower[i] == upper[i]
            return zero(atol)
        end
        min(sqrt(atol), (upper[i]-lower[i])/4)
    end
    lower, upper = lower+margin, upper-margin
    # The least-squares correction may violate a ratio cap.
    if all(method.lower .* p .< initial .< method.upper .* p) &&
       all(lower .- atol .<= A*initial .<= upper .+ atol)
        return initial, nothing
    end
    residual(q) = A*q .- clamp.(A*q, lower, upper)
    objective(q, _) = sum(abs2, residual(q))/2
    gradient!(g, q, _) = mul!(g, A', residual(q))
    function hessian!(h, q, _)
        value = A*q
        active = (lower .== upper) .| (value .< lower) .| (value .> upper)
        active_rows = A[active, :]
        mul!(h, active_rows', active_rows)
        nothing
    end
    f = Optimization.OptimizationFunction(
        objective,
        Optimization.NoAD();
        grad=gradient!,
        hess=hessian!,
    )
    problem = Optimization.OptimizationProblem(
        f,
        copy(p);
        lb=method.lower .* p,
        ub=method.upper .* p,
    )
    solution = Optimization.solve(problem, algorithm; kwargs...)
    collect(solution.u), solution
end

function _finite_independent_rows(A, lo, hi, atol)
    equalities = findall(lo .== hi)
    keep = Int[]
    for i in equalities
        row = vec(A[i, :])
        if isempty(keep)
            push!(keep, i)
            continue
        end
        coefficients = A[keep, :]' \ row
        residual = row - A[keep, :]'*coefficients
        if norm(residual) <= 32eps(eltype(A))*length(row)*max(1, norm(row))
            target_residual = lo[i] - dot(coefficients, lo[keep])
            lo[1]*minimum(residual)-atol <=
            target_residual <=
            lo[1]*maximum(residual)+atol ||
                throw(ArgumentError("infeasible linearly dependent expectation equalities"))
        else
            push!(keep, i)
        end
    end
    for i in findall(lo .!= hi)
        if all(iszero, view(A, i, :))
            lo[i] <= 0 <= hi[i] ||
                throw(ArgumentError("infeasible expectation bounds on the forced face"))
        else
            push!(keep, i)
        end
    end
    A[keep, :], lo[keep], hi[keep]
end

function _finite_residual(points, p, q, beliefs, method)
    ratio = q ./ p
    residual = max(
        abs(sum(q)-1),
        maximum(method.lower .- ratio),
        maximum(ratio .- method.upper),
        zero(eltype(q)),
    )
    for belief in beliefs
        h = Credence._tilt_feature(belief.statistic).(points)
        center = first(h)
        centered = h .- center
        scale = maximum(abs, centered)
        iszero(scale) && continue
        value = dot(q, centered)
        lower, upper = Credence._bounds(belief.relation)
        residual = max(residual, (lower-center-value)/scale, (value-(upper-center))/scale)
    end
    residual
end

function _finite_dual_gap(p, q, method, A, lower, upper, atol)
    ratio = q ./ p
    free = findall(r -> r > method.lower+atol && r < method.upper-atol, ratio)
    isempty(free) && return Inf
    active = findall(eachindex(lower)) do i
        value = dot(view(A, i, :), q)
        lower[i] == upper[i] || abs(value-lower[i]) <= atol || abs(value-upper[i]) <= atol
    end
    # Active rows can become dependent after restricting to uncapped coordinates.
    multipliers = pinv(A[active, free]') * log.(ratio[free])
    target = similar(multipliers)
    for (j, i) in enumerate(active)
        value = dot(view(A, i, :), q)
        if lower[i] == upper[i]
            target[j] = lower[i]
        elseif abs(value-lower[i]) <= atol
            multipliers[j] = max(zero(multipliers[j]), multipliers[j])
            target[j] = lower[i]
        else
            multipliers[j] = min(zero(multipliers[j]), multipliers[j])
            target[j] = upper[i]
        end
    end
    score = A[active, :]'*multipliers
    dual = dot(multipliers, target)
    for i in eachindex(p)
        r = exp(clamp(score[i], log(method.lower), log(method.upper)))
        phi = iszero(r) ? one(r) : r*log(r)-r+1
        dual -= p[i]*(score[i]*r-phi)
    end
    objective = _finite_entropy(q, p) - sum(q) + sum(p)
    max(zero(objective), objective-dual)
end

function Credence.tilt(
    reference::Credence.FiniteReference,
    rawbeliefs,
    method::Credence.BoundedKL,
    algorithm;
    atol=1e-8,
    kwargs...,
)
    isfinite(atol) && atol > 0 || throw(ArgumentError("atol must be finite and positive"))
    beliefs = Credence._tilt_beliefs(rawbeliefs)
    positive = findall(>(0), reference.weights)
    points = reference.points[positive]
    p = reference.weights[positive]
    function report(q, solver, optimality=:unresolved)
        prior, bounds, feasibility = nothing, (NaN, NaN), :unresolved
        normalization = Credence.Estimate(1)
        divergence = Credence.Estimate(Inf; evidence=:estimated)
        if q !== nothing
            q = max.(zero(eltype(q)), q)
            mass = sum(q)
            isfinite(mass) && mass > 0 || throw(
                ArgumentError("finite projection solver returned no finite positive mass"),
            )
            q ./= mass
            residual = _finite_residual(points, p, q, beliefs, method)
            if iszero(residual)
                feasibility = :feasible
            elseif residual <= atol
                feasibility = :within_tolerance
            else
                optimality = :unresolved
            end
            prior = Credence._finite_law(points, q)
            bounds = extrema(q ./ p)
            normalization = Credence.Estimate(mass)
            evidence = optimality == :analytic ? :analytic : :estimated
            divergence = Credence.Estimate(_finite_entropy(q, p); evidence)
            solver = (; solver..., residual, residual_tolerance=atol)
        end
        Credence._tilt_report(
            reference,
            beliefs,
            method,
            prior,
            solver,
            normalization,
            divergence,
            bounds,
            feasibility,
            optimality,
        )
    end
    H, lower, upper = _finite_rows(points, p, beliefs, atol)
    initial = copy(p)
    least, greatest = p, p
    fixed = falses(length(p))
    fixed_mass = zeros(eltype(p), length(p))
    lb, ub = method.lower .* p, method.upper .* p
    for i in axes(H, 1)
        row = vec(H[i, :])
        least, minmean = _finite_extreme(p, row, method, false)
        greatest, maxmean = _finite_extreme(p, row, method, true)
        max(minmean, lower[i]) <= min(maxmean, upper[i])+atol || throw(
            ArgumentError(
                "infeasible finite expectation: attainable interval [$minmean, $maxmean]",
            ),
        )
        extreme = if upper[i] == minmean
            least
        elseif lower[i] == maxmean
            greatest
        else
            nothing
        end
        if extreme !== nothing
            # At an extremal expectation, coordinates outside the partially filled
            # tied group are forced to their ratio caps in every feasible law.
            for j in eachindex(p)
                if extreme[j] == lb[j] || extreme[j] == ub[j]
                    fixed[j] = true
                    fixed_mass[j] = extreme[j]
                end
            end
        end
    end
    if all(lower .<= H*p .<= upper)
        return report(p, (; retcode=:Analytic), :analytic)
    end
    if size(H, 1) == 1
        h = vec(H)
        target = clamp(dot(p, h), only(lower), only(upper))
        q = if abs(target-dot(h, least)) <= 16eps(eltype(p))*max(1, abs(target))
            least
        elseif abs(target-dot(h, greatest)) <= 16eps(eltype(p))*max(1, abs(target))
            greatest
        elseif length(unique(h)) == 2
            lo, hi = extrema(h)
            above = h .== hi
            mass = (target-lo)/(hi-lo)
            ratios = ifelse.(above, mass/sum(p[above]), (1-mass)/sum(p[.!above]))
            p .* ratios
        else
            nothing
        end
        if q !== nothing
            return report(q, (; retcode=:Analytic), :analytic)
        end
        baseline = dot(p, h)
        extreme = target < baseline ? least : greatest
        weight = (target-baseline)/(dot(h, extreme)-baseline)
        initial .= (1-weight) .* p .+ weight .* extreme
        # For one feature, KL is minimized at the feasible mean nearest its reference mean.
        lower[1] = upper[1] = target
    end
    free = findall(!, fixed)
    isempty(free) && return report(fixed_mass, (; retcode=:Analytic), :analytic)
    remaining = 1-sum(fixed_mass)
    if remaining == sum(lb[free]) || remaining == sum(ub[free])
        fixed_mass[free] = remaining == sum(lb[free]) ? lb[free] : ub[free]
        return report(fixed_mass, (; retcode=:Analytic), :analytic)
    end
    A = vcat(ones(eltype(p), 1, length(p)), H)
    offset = A*fixed_mass
    lcons, ucons = vcat(one(eltype(p)), lower)-offset, vcat(one(eltype(p)), upper)-offset
    A, lcons, ucons = _finite_independent_rows(A[:, free], lcons, ucons, atol)
    equalities = findall(lcons .== ucons)
    if length(equalities) == length(free)
        q = copy(fixed_mass)
        q[free] = A[equalities, :] \ lcons[equalities]
        return report(q, (; retcode=:Analytic), :analytic)
    end
    initial, reduced_p = initial[free], p[free]
    start_solution = nothing
    objective(q, _) = _finite_entropy(q, reduced_p)
    gradient!(g, q, _) = (g .= log.(q ./ reduced_p) .+ 1)
    hessian!(h, q, _) = (h .= Diagonal(inv.(q)))
    constraints!(c, q, _) = mul!(c, A, q)
    jacobian!(j, q, _) = copyto!(j, A)
    f = Optimization.OptimizationFunction(
        objective,
        Optimization.NoAD();
        grad=gradient!,
        hess=hessian!,
        cons=constraints!,
        cons_j=jacobian!,
        cons_h=_finite_zero_hessians!,
    )
    solution = try
        if size(H, 1) > 1
            initial, start_solution = _finite_start(
                reduced_p,
                method,
                A,
                lcons,
                ucons,
                algorithm,
                atol;
                kwargs...,
            )
        end
        problem = Optimization.OptimizationProblem(
            f,
            initial;
            lb=lb[free],
            ub=ub[free],
            lcons,
            ucons,
        )
        Optimization.solve(problem, algorithm; kwargs...)
    catch error
        error isa InterruptException && rethrow()
        return report(nothing, (; retcode=:Failure, error, start_solution))
    end
    q = copy(fixed_mass)
    q[free] = solution.u
    if !all(isfinite, q) || any(<(0), q) || sum(q) <= 0
        return report(nothing, (; retcode=solution.retcode, solution, start_solution))
    end
    gap =
        all(>(0), q[free]) ?
        _finite_dual_gap(reduced_p, (q/sum(q))[free], method, A, lcons, ucons, atol) : Inf
    optimality = isfinite(gap) && gap <= atol ? :numerical : :unresolved
    report(
        q,
        (;
            retcode=solution.retcode,
            solution,
            start_solution,
            dual_gap=gap,
            derivatives=:analytic,
        ),
        optimality,
    )
end
