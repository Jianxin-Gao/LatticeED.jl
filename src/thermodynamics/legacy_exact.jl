"""
    thermal_expectation(energies, vectors, O, beta; mode=:op, state=nothing)

Exact expectation from an orthonormal eigensystem. In :state mode, state is a
vector or a matrix of state columns, and exp(-beta*H/2) acts on each column.
Boltzmann shifts cancel in the ratio; no matrix inverse is formed.
"""
function thermal_expectation(energies::AbstractVector{<:Real}, vectors::AbstractMatrix,
                             O::AbstractMatrix, beta; mode=:op, state=nothing)
    bs = _checked_beta(beta)
    n = length(energies)
    n > 0 && size(vectors) == size(O) == (n,n) || throw(DimensionMismatch("invalid eigensystem or observable"))
    all(isfinite, energies) || throw(ArgumentError("energies must be finite"))
    mode in (:op,:state) || throw(ArgumentError("mode must be :op or :state"))
    shifted = energies .- minimum(energies)
    result = zeros(ComplexF64, length(bs))
    if mode == :op
        work = Vector{promote_type(eltype(vectors),eltype(O))}(undef, n)
        diagonal = [dot(view(vectors,:,i), mul!(work,O,view(vectors,:,i))) for i in 1:n]
        for (i,b) in enumerate(bs)
            weights = exp.(-b .* shifted)
            result[i] = dot(weights, diagonal) / sum(weights)
        end
    else
        state === nothing && throw(ArgumentError("state is required for mode=:state"))
        size(state,1) == n || throw(DimensionMismatch("state has the wrong dimension"))
        amplitudes = vectors' * state
        populations = state isa AbstractVector ? abs2.(amplitudes) : vec(sum(abs2,amplitudes;dims=2))
        supported = findall(>(0),populations)
        isempty(supported) && throw(ArgumentError("state must be nonzero"))
        shifted = energies .- minimum(energies[supported])
        factors = zeros(n)
        for (i,b) in enumerate(bs)
            for j in supported
                factors[j] = exp(-b/2 * shifted[j])
            end
            evolved = vectors * (factors .* amplitudes)
            norm2 = sum(abs2, evolved)
            norm2 > 0 || throw(ArgumentError("thermal state has zero norm"))
            result[i] = dot(evolved, O * evolved) / norm2
        end
    end
    return result
end
