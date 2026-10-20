"""Small, serializable Lanczos record. No Hilbert-space vectors or operators are retained."""
struct LanczosSample
    key::String
    dim::Int
    multiplicity::Int
    particles::Float64
    physical_shift::Float64
    energy_offset::Float64
    model_id::String
    sample_id::Int
    sampling::Symbol
    seed::UInt64
    requested_M::Int
    reorthogonalize::Bool
    tol::Float64
    alpha::Vector{Float64}
    beta::Vector{Float64}
    energies::Vector{Float64}
    weights::Vector{Float64}
    metadata::NamedTuple
end

"""Stable SHA-derived seed; excludes M, beta, thread count and scheduling order."""
function sample_seed(seed::Integer, key, sample_id::Integer)
    seed >= 0 && sample_id > 0 || throw(ArgumentError("invalid sample seed/id"))
    return parse(UInt64, _digest((string(seed), string(key), sample_id))[1:16]; base=16)
end

function _lanczos_sample(block::ThermalBlock, H, id; M, seed, sampling,
                         reorthogonalize, tol, rng=nothing, metadata=NamedTuple())
    size(H) == (block.dim, block.dim) || throw(DimensionMismatch("block factory has wrong dimension"))
    s = rng === nothing ? sample_seed(seed, block.model_id * ":" * block.key, id) : rand(rng, UInt64)
    T = promote_type(Float64, eltype(H))
    timed = @timed begin
        if sampling == :basis
            v = zeros(T, block.dim)
            v[id] = one(T)
        else
            v = randn(Xoshiro(s), T, block.dim)
        end
        lz = lanczos(H, v; m=M, tol, reorthogonalize, store_vectors=reorthogonalize)
        # Small dense projection avoids LAPACK stegr failures for nearly repeated
        # Ritz values after loss of orthogonality in three-vector Lanczos.
        eig = eigen(Symmetric(Matrix(SymTridiagonal(lz.α, lz.β))))
        (lz.α, lz.β, eig.values, abs2.(eig.vectors[1, :]))
    end
    a, b, es, ws = timed.value
    meta = merge(metadata, (elapsed_s=timed.time, allocation_bytes=timed.bytes,
                            gc_s=timed.gctime, krylov_dim=length(a), threads=Threads.nthreads(),
                            julia_version=string(VERSION)))
    return LanczosSample(block.key, block.dim, block.multiplicity, block.particles,
        block.physical_shift, block.energy_offset, block.model_id, id, sampling, s,
        M, reorthogonalize, Float64(tol), a, b, es, ws, meta)
end

"""
    ftlm_samples(blocks; R=32, M=64, seed=1234, sampling=:random,
                 reorthogonalize=false, tol=1e-12, verbose=false)

Generate temperature-independent records, one sector at a time. :basis uses
every canonical basis vector and never diagonalizes the original Hamiltonian.
An optional rng supports legacy callers; checkpoints use deterministic seeds.
"""
function ftlm_samples(blocks::AbstractVector; R::Int=32, M::Int=64, seed::Integer=1234,
                      sampling::Symbol=:random, reorthogonalize::Bool=false,
                      tol::Real=1e-12, verbose::Bool=false, rng=nothing)
    _check_sampling(R, M, seed, sampling, tol)
    bs = _as_thermal_blocks(blocks)
    length(unique(b.key for b in bs)) == length(bs) || throw(ArgumentError("duplicate block keys"))
    metadata = (expected_block_keys=sort([b.key for b in bs]),
                trace_id=_digest(([(b.key, b.dim, b.multiplicity, b.particles, b.physical_shift,
                                   b.energy_offset, b.model_id) for b in bs],
                                  M, string(seed), sampling, reorthogonalize, tol)))
    records = LanczosSample[]
    for (iblock, block) in enumerate(bs)
        verbose && @info "Building sector" sector="$(iblock)/$(length(bs))" key=block.key dimension=block.dim
        H = block.build()
        release = _large_operator(H)
        started = time_ns()
        count = sampling == :basis ? block.dim : R
        verbose && @info "Sampling" sector=block.key dimension=block.dim samples=count M
        for id in 1:count
            sample = _lanczos_sample(block, H, id; M, seed, sampling, reorthogonalize, tol, rng, metadata)
            push!(records, sample)
            verbose && _log_sample(sample, count)
        end
        H = nothing
        # Avoid retaining a previous block's diagonal alongside the next factory allocation.
        verbose && @info "Sector complete" sector=block.key elapsed_s=(time_ns()-started)/1e9
        release && GC.gc(true)
    end
    return records
end

function ftlm_samples(H; dim::Int=size(H, 1), kwargs...)
    return ftlm_samples([ThermalBlock("1", dim, () -> H)]; kwargs...)
end
