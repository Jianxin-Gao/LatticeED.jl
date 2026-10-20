function _checked_beta(beta)
    bs = beta isa Real ? [Float64(beta)] : Float64.(collect(beta))
    !isempty(bs) && all(b -> isfinite(b) && b >= 0, bs) || throw(ArgumentError("beta must be nonempty, finite and nonnegative"))
    return bs
end

function _sample_groups(samples)
    isempty(samples) && throw(ArgumentError("no thermal samples"))
    groups = Dict{String,Vector{Int}}()
    seen = Set{Tuple{String,Int}}()
    for (i, s) in enumerate(samples)
        id = (s.key, s.sample_id)
        id in seen && throw(ArgumentError("duplicate sample $id"))
        push!(seen, id)
        push!(get!(groups, s.key, Int[]), i)
        all(isfinite, s.energies) && all(w -> isfinite(w) && w >= 0, s.weights) &&
            length(s.energies) == length(s.weights) && isapprox(sum(s.weights), 1; atol=1e-10) ||
            throw(ArgumentError("invalid spectral record"))
    end
    for ids in values(groups)
        a = samples[first(ids)]
        for i in ids
            s = samples[i]
            (s.dim, s.multiplicity, s.model_id, s.sampling, s.requested_M, s.reorthogonalize,
             s.tol, s.energy_offset, s.particles, s.physical_shift) ==
            (a.dim, a.multiplicity, a.model_id, a.sampling, a.requested_M, a.reorthogonalize,
             a.tol, a.energy_offset, a.particles, a.physical_shift) ||
                throw(ArgumentError("incompatible records for block $(a.key)"))
        end
        a.sampling == :basis && sort([samples[i].sample_id for i in ids]) != collect(1:a.dim) &&
            throw(ArgumentError("basis trace is incomplete"))
    end
    for name in (:fingerprint, :trace_id, :expected_block_keys)
        entries = [get(s.metadata, name, nothing) for s in samples]
        all(==(first(entries)), entries) || throw(ArgumentError("incompatible sample collection metadata: $name"))
    end
    expected = get(first(samples).metadata, :expected_block_keys, nothing)
    expected === nothing || sort(collect(keys(groups))) == expected ||
        throw(ArgumentError("thermal trace is missing entire blocks"))
    return groups
end

function _stratified_errors(z, numer, groups, samples)
    nβ = size(z, 2)
    variance = zeros(nβ)
    ztotal, ntotal = vec(sum(z; dims=1)), vec(sum(numer; dims=1))
    for ids in values(groups)
        samples[first(ids)].sampling == :random || continue
        n = length(ids)
        n > 1 || return fill(NaN, nβ)
        bz, bn = vec(sum(view(z, ids, :); dims=1)), vec(sum(view(numer, ids, :); dims=1))
        estimates = Vector{Float64}(undef, n)
        for j in 1:nβ
            for (k,i) in enumerate(ids)
                estimates[k] = (ntotal[j] - bn[j] + (bn[j] - numer[i,j]) * n / (n-1)) /
                               (ztotal[j] - bz[j] + (bz[j] - z[i,j]) * n / (n-1))
            end
            avg = sum(estimates) / n
            variance[j] += (n-1) / n * sum(x -> abs2(x-avg), estimates)
        end
    end
    return sqrt.(variance)
end

function _thermal_weights(samples, beta)
    bs = _checked_beta(beta)
    groups = _sample_groups(samples)
    logs = Matrix{Float64}(undef, length(samples), length(bs))
    means = similar(logs)
    for ids in values(groups), i in ids
        s = samples[i]
        energies = s.energies .- s.energy_offset
        lw = [w == 0 ? -Inf : log(w) for w in s.weights]
        prefactor = log(s.dim) + log(s.multiplicity) - log(length(ids))
        exponents, weights = similar(lw), similar(lw)
        for (j, b) in enumerate(bs)
            @. exponents = lw - b * energies
            scale = maximum(exponents)
            @. weights = exp(exponents - scale)
            z = sum(weights)
            logs[i, j] = prefactor + scale + log(z)
            means[i, j] = dot(weights, energies) / z
        end
    end
    scales = vec(maximum(logs; dims=1))
    z = exp.(logs .- scales')
    return (β=bs, groups=groups, z=z, means=means, scales=scales)
end

_sample_signature(samples) = _digest([(s.key,s.dim,s.multiplicity,s.particles,s.physical_shift,
    s.energy_offset,s.model_id,s.sample_id,s.sampling,s.seed,s.requested_M,s.reorthogonalize,
    s.tol,s.energies,s.weights) for s in samples])

"""
    thermal_energy(samples, beta; nsites=1)

Stable canonical/grand-canonical traces. values is total <K>; energy_per_site
is physical <H0>/nsites. Numerical offsets are subtracted before weighting.
Jackknife errors exclude Krylov bias. logZ is authoritative; compatibility
Z/numerator can overflow and are never used to compute ratios.
"""
function thermal_energy(samples::AbstractVector{LanczosSample}, beta; nsites::Int=1)
    nsites > 0 || throw(ArgumentError("nsites must be positive"))
    weights = _thermal_weights(samples, beta)
    bs, groups, z, means, scales = weights.β, weights.groups, weights.z, weights.means, weights.scales
    k = z .* means
    p = z .* (means .+ [s.physical_shift for s in samples])
    number = z .* [s.particles for s in samples]
    denom = vec(sum(z; dims=1))
    logZ = scales .+ log.(denom)
    kvals = vec(sum(k; dims=1)) ./ denom
    physical = vec(sum(p; dims=1)) ./ denom
    particles = vec(sum(number; dims=1)) ./ denom
    errors = _stratified_errors(z, k, groups, samples)
    physical_errors = all(s -> iszero(s.physical_shift), samples) ? errors : _stratified_errors(z, p, groups, samples)
    number_errors = _stratified_errors(z, number, groups, samples)
    trace_dim = sum(big(samples[first(ids)].dim) * samples[first(ids)].multiplicity for ids in values(groups))
    return (β=bs, values=kvals, errors=errors, physical_values=physical,
        physical_errors=physical_errors, energy_per_site=physical ./ nsites,
        energy_errors=physical_errors ./ nsites, grand_energy_per_site=kvals ./ nsites,
        grand_energy_errors=errors ./ nsites, particles=particles, particle_errors=number_errors,
        density=particles ./ nsites, density_errors=number_errors ./ nsites,
        logZ=logZ, Z=exp.(logZ), numerator=kvals .* exp.(logZ),
        trace_dim=trace_dim, log_scale=scales, sample_Z_scaled=z,
        sample_numerator_scaled=k, sample_signature=_sample_signature(samples),
        error_definition="stratified jackknife of the full trace ratio; excludes Krylov bias")
end

function ftlm_energy(blocks::AbstractVector, beta; nsites::Int=1, kwargs...)
    return thermal_energy(ftlm_samples(blocks; kwargs...), beta; nsites)
end
function ftlm_energy(H, beta; dim::Int=size(H,1), nsites::Int=1, kwargs...)
    return thermal_energy(ftlm_samples(H; dim, kwargs...), beta; nsites)
end

"""
    thermal_block_observable(samples, beta, values; normalization=1)

Validated special case: an observable is a known scalar times identity in each
sector (for example total Sz). `values` maps every block key to that scalar.
This does not approximate a general noncommuting observable.
"""
function thermal_block_observable(samples, beta, values::AbstractDict; normalization::Real=1)
    weights = _thermal_weights(samples, beta)
    return _block_ratio(samples, weights.β, weights.z, weights.groups, values, normalization)
end

"""Reuse an existing thermal_energy result without repeating spectral weights or energy statistics."""
function thermal_block_observable(samples, energy::NamedTuple, values::AbstractDict; normalization::Real=1)
    energy.sample_signature == _sample_signature(samples) || throw(ArgumentError("energy cache belongs to different samples"))
    return _block_ratio(samples, energy.β, energy.sample_Z_scaled, _sample_groups(samples), values, normalization)
end

function _block_ratio(samples, beta, z, groups, values, normalization)
    isfinite(normalization) && normalization > 0 || throw(ArgumentError("invalid normalization"))
    scalars = [Float64(values[s.key]) for s in samples]
    all(isfinite, scalars) || throw(ArgumentError("block observable must be finite"))
    numerator = z .* scalars
    result = vec(sum(numerator; dims=1)) ./ vec(sum(z; dims=1)) ./ normalization
    errors = _stratified_errors(z, numerator, groups, samples) ./ normalization
    return (β=beta, values=result, errors=errors)
end

"""Guarded dense reference. Never called implicitly by ftlm_samples or ftlm_energy."""
function exact_samples(blocks::AbstractVector; maxdim::Int=4096, verbose::Bool=false)
    bs = _as_thermal_blocks(blocks)
    all(b -> b.dim <= maxdim, bs) || throw(ArgumentError("exact block exceeds maxdim=$maxdim"))
    length(unique(b.key for b in bs)) == length(bs) || throw(ArgumentError("duplicate block keys"))
    expected = sort([b.key for b in bs])
    samples = LanczosSample[]
    for b in bs
        verbose && @info "Diagonalizing" sector=b.key dimension=b.dim
        started = time_ns()
        H = b.build()
        dense = Matrix(H)
        ishermitian(dense) || throw(ArgumentError("exact_energy requires a Hermitian Hamiltonian"))
        es = eigvals(Hermitian(dense))
        push!(samples, LanczosSample(b.key, b.dim, b.multiplicity, b.particles,
            b.physical_shift, b.energy_offset, b.model_id, 1, :exact, UInt64(0), b.dim,
            true, 0.0, Float64[], Float64[], es, fill(1 / b.dim, b.dim),
            (method="dense exact", expected_block_keys=expected)))
        verbose && @info "Sector complete" sector=b.key minimum_energy=minimum(es)-b.energy_offset elapsed_s=(time_ns()-started)/1e9
    end
    return samples
end

"""Guarded exact energy, using the same aggregation as FTLM."""
exact_energy(blocks::AbstractVector, beta; nsites::Int=1, maxdim::Int=4096, verbose::Bool=false) =
    thermal_energy(exact_samples(blocks; maxdim, verbose), beta; nsites)
