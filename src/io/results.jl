Base.@kwdef struct EDResult
    parameters::NamedTuple = NamedTuple()
    model::NamedTuple = NamedTuple()
    lattice::NamedTuple = NamedTuple()
    exact::NamedTuple = NamedTuple()
    observables::NamedTuple = NamedTuple()
    metadata::NamedTuple = NamedTuple()
end

function save_result(path::AbstractString, result; varname::Symbol=:result)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    temporary, io = mktemp(isempty(dir) ? "." : dir)
    close(io)
    try
        JLD2.jldopen(temporary, "w") do file
            file[string(varname)] = result
        end
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
    return path
end

function load_result(path::AbstractString; varname::Symbol=:result)
    # Files written before the package rename retain the original module path.
    typemap = Dict("ED_jl.EDResult" => EDResult,
                   "ED_jl.LanczosSample" => LanczosSample)
    return JLD2.load(path, string(varname); typemap)
end
