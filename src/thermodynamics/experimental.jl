# Experimental observable estimator. Noncommuting observables and low-temperature
# overflow are deliberately outside the validated energy backend.
function _jackknife_ratio_errors(z, numerator)
    R, count = size(z)
    R > 1 || return fill(NaN, count)
    zs, ns = vec(sum(z; dims=1)), vec(sum(numerator; dims=1))
    return map(1:count) do j
        estimates = [(ns[j]-numerator[i,j])/(zs[j]-z[i,j]) for i in 1:R]
        average = sum(estimates)/R
        sqrt((R-1)/R * sum(x -> abs2(x-average), estimates))
    end
end

function _block_observable(block)
    block isa NamedTuple && haskey(block,:O) && return block.O
    throw(ArgumentError("observable blocks require an O field"))
end

function _observable_sample!(z, numerator, H, O, beta; rng, dim, M, tol, reorthogonalize)
    vector = randn(rng, ComplexF64, dim)
    lz = lanczos(H, vector; m=M, tol, reorthogonalize, store_vectors=true)
    projection = eigen(Symmetric(Matrix(SymTridiagonal(lz.α,lz.β))))
    weights = abs2.(projection.vectors[1,:])
    states = lz.V * projection.vectors
    work = similar(view(states,:,1))
    diagonal = [dot(view(states,:,i),mul!(work,O,view(states,:,i))) for i in axes(states,2)]
    emin = minimum(projection.values)
    for (i,b) in enumerate(beta)
        thermal_weights = weights .* exp.(-b .* (projection.values .- emin))
        scale = exp(-b*emin)
        z[i] = scale * sum(thermal_weights)
        numerator[i] = scale * sum(thermal_weights .* diagonal)
    end
    return length(lz.α)
end

"""
    ftlm_expectation(H, O, beta; R=32, M=64, ...)

Experimental general observable estimate. For O === H, dispatch to the validated
energy backend. General noncommuting observables and low-temperature overflow
remain unvalidated. Log settings do not change the calculation or RNG sequence.
"""
function ftlm_expectation(H, O, beta; R::Int=32, M::Int=64,
        rng::AbstractRNG=Random.default_rng(), dim::Int=size(H,1), tol::Real=1e-12,
        reorthogonalize::Bool=true, verbose::Bool=false, _energy_fastpath::Bool=true)
    _energy_fastpath && O === H && return ftlm_energy(H,beta; R,M,rng,dim,tol,reorthogonalize,verbose)
    _check_sampling(R,M,0,:random,tol)
    size(H) == size(O) == (dim,dim) || throw(DimensionMismatch("observable dimensions"))
    bs = _checked_beta(beta)
    z, numerator = zeros(R,length(bs)), zeros(ComplexF64,R,length(bs))
    verbose && @warn "Experimental observable estimator" low_temperature_overflow="not validated"
    for sample in 1:R
        timed = @timed _observable_sample!(view(z,sample,:),view(numerator,sample,:),H,O,bs;
            rng,dim,M,tol,reorthogonalize)
        verbose && @info "Observable sample complete" sample samples=R krylov_dim=timed.value elapsed_s=timed.time allocation_bytes=timed.bytes gc_s=timed.gctime
    end
    z .*= dim
    numerator .*= dim
    Z, N = vec(sum(z;dims=1))./R, vec(sum(numerator;dims=1))./R
    return (β=bs, values=N./Z, errors=_jackknife_ratio_errors(z,numerator), Z=Z,
        numerator=N, samples=R, krylov_dim=M, trace_dim=dim,
        sample_Z=z, sample_numerator=numerator)
end

"""Experimental multi-block observable estimate; blocks carry H, O and dim or basis."""
function ftlm_expectation(blocks::AbstractVector, beta; R::Int=32, M::Int=64,
        rng::AbstractRNG=Random.default_rng(), tol::Real=1e-12,
        reorthogonalize::Bool=true, verbose::Bool=false)
    isempty(blocks) && throw(ArgumentError("blocks must not be empty"))
    bs = _checked_beta(beta)
    z, numerator = zeros(R,length(bs)), zeros(ComplexF64,R,length(bs))
    dims = Int[]
    for block in blocks
        dim = _block_dim(block)
        push!(dims,dim)
        result = ftlm_expectation(_block_hamiltonian(block),_block_observable(block),bs;
            R,M,rng,dim,tol,reorthogonalize,verbose,_energy_fastpath=false)
        z .+= result.sample_Z
        numerator .+= result.sample_numerator
    end
    Z, N = vec(sum(z;dims=1))./R, vec(sum(numerator;dims=1))./R
    return (β=bs, values=N./Z, errors=_jackknife_ratio_errors(z,numerator), Z=Z,
        numerator=N, samples_per_block=R, krylov_dim=M, trace_dim=sum(dims),
        block_dims=dims, sample_Z=z, sample_numerator=numerator)
end
