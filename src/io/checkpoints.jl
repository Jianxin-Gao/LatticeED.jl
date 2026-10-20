const SAMPLE_FORMAT_VERSION = 1

function _source_content(root)
    paths = String[]
    for (dir, _, files) in walkdir(joinpath(root, "src")), file in files
        endswith(file, ".jl") && push!(paths, relpath(joinpath(dir, file), root))
    end
    isfile(joinpath(root, "Project.toml")) && push!(paths, "Project.toml")
    return _digest([(p, read(joinpath(root, p), String)) for p in sort!(paths)])
end

function _dependency_metadata()
    packages = Pkg.dependencies()
    own_uuid = Base.PkgId(@__MODULE__).uuid
    roots = haskey(packages, own_uuid) ? packages[own_uuid].dependencies : Pkg.project().dependencies
    pending, seen = collect(values(roots)), Set{Base.UUID}()
    records = NamedTuple[]
    while !isempty(pending)
        uuid = pop!(pending)
        uuid in seen && continue
        push!(seen, uuid)
        package = packages[uuid]
        append!(pending, values(package.dependencies))
        content = package.is_tracking_path ? _source_content(package.source) : nothing
        push!(records, (name=package.name, uuid=string(uuid), version=string(package.version),
                       tree_hash=package.tree_hash, source_content=content))
    end
    return sort!(records; by=r -> r.uuid)
end

_source_fingerprint(dependencies) = _digest((_source_content(normpath(joinpath(@__DIR__, "..", ".."))), dependencies))

"""Digest source and resolved dependency identities, independent of local Manifest formatting."""
source_fingerprint() = _source_fingerprint(_dependency_metadata())

function _run_identity(blocks; M, seed, sampling, reorthogonalize, tol)
    all(b -> !isempty(b.model_id), blocks) ||
        throw(ArgumentError("checkpoint factories require an explicit model_id containing all model parameters"))
    descriptors = sort([(b.key, b.dim, b.multiplicity, b.particles, b.physical_shift,
                         b.energy_offset, b.model_id, b.members) for b in blocks]; by=first)
    source = source_fingerprint()
    id = _digest((SAMPLE_FORMAT_VERSION, source, string(VERSION), descriptors,
                  M, string(seed), sampling, reorthogonalize, Float64(tol)))
    return (fingerprint=id, source_fingerprint=source, format_version=SAMPLE_FORMAT_VERSION,
            expected_block_keys=sort([b.key for b in blocks]))
end

_sample_path(dir, identity, key, id) = joinpath(dir, identity.fingerprint, _digest(key), "sample_$(id).jld")

function save_sample(path::AbstractString, sample::LanczosSample)
    return save_result(path, (format_version=SAMPLE_FORMAT_VERSION, sample=sample); varname=:checkpoint)
end

function _load_sample(path, identity, b, id; M, seed, sampling, reorthogonalize, tol)
    envelope = load_result(path; varname=:checkpoint)
    envelope.format_version == SAMPLE_FORMAT_VERSION || throw(ArgumentError("unsupported checkpoint version: $path"))
    s = envelope.sample
    expected = (b.key, b.dim, b.multiplicity, b.particles, b.physical_shift, b.energy_offset,
                b.model_id, id, sampling, sample_seed(seed, b.model_id * ":" * b.key, id),
                M, reorthogonalize, Float64(tol))
    actual = (s.key, s.dim, s.multiplicity, s.particles, s.physical_shift, s.energy_offset,
              s.model_id, s.sample_id, s.sampling, s.seed, s.requested_M, s.reorthogonalize, s.tol)
    actual == expected && s.metadata.fingerprint == identity.fingerprint &&
        s.metadata.source_fingerprint == identity.source_fingerprint &&
        s.metadata.julia_version == string(VERSION) || throw(ArgumentError("incompatible checkpoint: $path"))
    return s
end

function _check_sampling(R, M, seed, sampling, tol)
    R > 0 && M > 0 && seed >= 0 || throw(ArgumentError("invalid R, M or seed"))
    sampling in (:random, :basis) || throw(ArgumentError("invalid sampling method"))
    isfinite(tol) && tol >= 0 || throw(ArgumentError("invalid tolerance"))
end

"""
    run_ftlm!(directory, blocks; R=4, M=64, seed=1234, shard=0, nshards=1, ...)

Atomic per-sample checkpoints, with zero-based independent task sharding.
Changing R/beta/thread count reuses samples. M/mode/model/code changes are isolated
in separate content-addressed run directories. Existing samples are validated.
"""
function run_ftlm!(dir::AbstractString, blocks::AbstractVector; R::Int=4, M::Int=64,
                   seed::Integer=1234, sampling::Symbol=:random, reorthogonalize::Bool=false,
                   tol::Real=1e-12, shard::Int=0, nshards::Int=1, verbose::Bool=true)
    _check_sampling(R, M, seed, sampling, tol)
    0 <= shard < nshards || throw(ArgumentError("require 0 <= shard < nshards"))
    bs = _as_thermal_blocks(blocks)
    length(unique(b.key for b in bs)) == length(bs) || throw(ArgumentError("duplicate block keys"))
    identity = _run_identity(bs; M, seed, sampling, reorthogonalize, tol)
    saved, skipped, task = 0, 0, 0
    for b in bs
        todo = Int[]
        for id in 1:(sampling == :basis ? b.dim : R)
            assigned = mod(task, nshards) == shard
            task += 1
            assigned || continue
            path = _sample_path(dir, identity, b.key, id)
            if isfile(path)
                _load_sample(path, identity, b, id; M, seed, sampling, reorthogonalize, tol)
                skipped += 1
            else
                push!(todo, id)
            end
        end
        isempty(todo) && continue
        verbose && @info "Sampling" sector=b.key samples=length(todo) dimension=b.dim
        H = b.build()
        release = _large_operator(H)
        started = time_ns()
        for id in todo
            record = _lanczos_sample(b, H, id; M, seed, sampling, reorthogonalize, tol, metadata=identity)
            save_sample(_sample_path(dir, identity, b.key, id), record)
            saved += 1
            verbose && _log_sample(record, sampling == :basis ? b.dim : R)
        end
        H = nothing
        verbose && @info "Sector complete" sector=b.key elapsed_s=(time_ns()-started)/1e9 saved=length(todo)
        release && GC.gc(true)
    end
    return merge(identity, (saved=saved, skipped=skipped, run_directory=joinpath(dir, identity.fingerprint)))
end

"""Load exactly the requested complete trace; missing, duplicate or incompatible records fail."""
function load_samples(dir::AbstractString, blocks::AbstractVector; R::Int=4, M::Int=64,
                      seed::Integer=1234, sampling::Symbol=:random,
                      reorthogonalize::Bool=false, tol::Real=1e-12)
    _check_sampling(R, M, seed, sampling, tol)
    bs = _as_thermal_blocks(blocks)
    identity = _run_identity(bs; M, seed, sampling, reorthogonalize, tol)
    records = LanczosSample[]
    for b in bs, id in 1:(sampling == :basis ? b.dim : R)
        path = _sample_path(dir, identity, b.key, id)
        isfile(path) || throw(ArgumentError("missing checkpoint for block $(b.key), sample $id, M=$M"))
        push!(records, _load_sample(path, identity, b, id; M, seed, sampling, reorthogonalize, tol))
    end
    _sample_groups(records)
    return records
end
