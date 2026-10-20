mutable struct RunLogger <: AbstractLogger
    file::IO
    console::Union{IO,Nothing}
    started::UInt64
    stage::String
    log_every::Int
    lock::ReentrantLock
end

Logging.min_enabled_level(::RunLogger) = Logging.Info
Logging.shouldlog(::RunLogger, level, _module, group, id) = level >= Logging.Info
Logging.catch_exceptions(::RunLogger) = false

_log_value(x::AbstractFloat) = isfinite(x) ? @sprintf("%.8g", x) : string(x)
_log_value(x::AbstractString) = x
_log_value(x) = repr(x)
_parameter_value(x) = x isa AbstractString ? x : repr(x)

function Logging.handle_message(logger::RunLogger, level, message, _module, group, id, file, line; kwargs...)
    lock(logger.lock) do
        logger.stage = string(message)
        if message == "Sample complete"
            sample, count = kwargs[:sample], kwargs[:samples]
            sample == 1 || sample == count || sample % logger.log_every == 0 || return
        end
        elapsed = (time_ns() - logger.started) / 1e9
        seconds = @sprintf("%05.2f", rem(elapsed, 60))
        prefix = @sprintf("[%02d:%02d:%s] ", floor(Int, elapsed / 3600), floor(Int, elapsed / 60) % 60, seconds)
        fields = ["$(key)=$(_log_value(value))" for (key, value) in kwargs if key != :exception]
        text = prefix * (level >= Logging.Warn ? string(level) * " | " : "") * string(message)
        isempty(fields) || (text *= " | " * join(fields, " | "))
        for io in (logger.file, logger.console)
            io === nothing && continue
            println(io, text)
            if haskey(kwargs, :exception)
                err, bt = kwargs[:exception]
                showerror(io, err, bt)
                println(io)
            end
            flush(io)
        end
    end
end

function _print_parameters(io, parameters; indent="")
    for (key, value) in pairs(parameters)
        if value isa NamedTuple || value isa AbstractDict
            println(io, indent, key, ":")
            _print_parameters(io, value; indent=indent * "  ")
        elseif value isa AbstractVector && !isempty(value) && all(v -> v isa NamedTuple && haskey(v,:name),value)
            println(io, indent, key, " (", length(value), " entries):")
            for entry in sort(value;by=v->v.name)
                fields = ["$(k)=$(_parameter_value(v))" for (k,v) in pairs(entry) if k != :name && v !== nothing]
                println(io, indent, "  ", entry.name, " | ", join(fields," | "))
            end
        elseif value isa AbstractVector && length(value) > 12
            println(io, indent, key, " (", length(value), " entries):")
            for start in 1:12:length(value)
                println(io, indent, "  ", join(_parameter_value.(value[start:min(start+11, end)]), ", "))
            end
        else
            println(io, indent, key, ": ", _parameter_value(value))
        end
    end
end

"""Reproducibility metadata for the loaded package and its dependency closure."""
function run_metadata()
    dependencies = _dependency_metadata()
    return (package_version=string(pkgversion(@__MODULE__)), julia_version=string(VERSION),
        source_fingerprint=_source_fingerprint(dependencies), dependencies=dependencies,
        threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads(),
        started_utc=string(now(UTC), "Z"))
end

"""
    with_run_log(f, path; parameters, metadata=run_metadata(), console=true, log_every=1)

Write complete parameters before calling `f()`. Capture scoped Julia log events
in a human-readable file and optionally stderr. Flush each event, record failures
with a backtrace, and rethrow them. Never replace the process-wide logger.
"""
function with_run_log(f, path::AbstractString; parameters, metadata=run_metadata(),
                      console::Bool=true, log_every::Int=1)
    log_every > 0 || throw(ArgumentError("log_every must be positive"))
    mkpath(dirname(abspath(path)))
    return open(path, "w") do file
        logger = RunLogger(file, console ? stderr : nothing, time_ns(), "Initialization", log_every, ReentrantLock())
        for io in (file, logger.console)
            io === nothing && continue
            println(io, "LatticeED run\nParameters:")
            _print_parameters(io, parameters; indent="  ")
            println(io, "Environment:")
            _print_parameters(io, metadata; indent="  ")
            println(io, "Error estimate: stochastic trace sampling only; excludes Krylov bias.\n")
            flush(io)
        end
        with_logger(logger) do
            try
                result = f()
                @info "Completed" elapsed_s=(time_ns()-logger.started)/1e9
                return result
            catch err
                @error "Failed" stage=logger.stage output_directory=dirname(abspath(path)) exception=(err, catch_backtrace())
                rethrow()
            end
        end
    end
end

_estimate(value, error) = "$(_log_value(value)) ± $(isnan(error) ? "not estimable (R=1)" : _log_value(error))"

"""Log representative temperature points; the result file retains the full grid."""
function log_thermal_results(energy; observables=NamedTuple())
    for i in unique([1, cld(length(energy.β), 2), length(energy.β)])
        extra = (; (key => _estimate(value.values[i], value.errors[i]) for (key, value) in pairs(observables))...)
        @info "Results" beta=energy.β[i] energy_per_site=_estimate(energy.energy_per_site[i], energy.energy_errors[i]) grand_energy_per_site=_estimate(energy.grand_energy_per_site[i], energy.grand_energy_errors[i]) logZ=energy.logZ[i] extra...
    end
end

function _log_sample(sample, count)
    @info "Sample complete" sector=sample.key sample=sample.sample_id samples=count krylov="$(length(sample.alpha))/$(sample.requested_M)" ritz_minimum_estimate=minimum(sample.energies)-sample.energy_offset elapsed_s=sample.metadata.elapsed_s allocation_bytes=sample.metadata.allocation_bytes gc_s=sample.metadata.gc_s
end

# Retain bounded memory at large sector boundaries without collecting every sample.
_large_operator(H) = Base.summarysize(H) >= 256 * 2^20
