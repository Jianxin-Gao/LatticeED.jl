"""
    compare_convergence(results; atol=nothing)

Compare adjacent energy estimates on the same beta grid. Report changes and
stochastic errors separately; neither establishes a bound on Krylov bias.
"""
function compare_convergence(results::AbstractVector; atol=nothing)
    length(results) >= 2 || throw(ArgumentError("need at least two estimates"))
    atol === nothing || (isfinite(atol) && atol >= 0) || throw(ArgumentError("invalid tolerance"))
    return map(2:length(results)) do i
        a, b = results[i-1], results[i]
        a.β == b.β || throw(ArgumentError("beta grids differ"))
        delta = maximum(abs.(a.energy_per_site .- b.energy_per_site))
        error = maximum(b.energy_errors)
        status = atol === nothing ? "assessment_required" :
            isfinite(error) && delta <= atol && error <= atol ? "meets_requested_checks" : "not_converged"
        (from=i-1, to=i, max_energy_change=delta, max_stochastic_error=error,
         status=status, qualification="M changes and stochastic errors are separate diagnostics")
    end
end
