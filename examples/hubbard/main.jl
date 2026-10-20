if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(joinpath(@__DIR__,".."))
end

module HubbardExample
using LatticeED
include("config.jl")

function main(config=example_config())
    mkpath(config.output.root)
    directory = mktempdir(config.output.root; prefix="run-", cleanup=false)
    log_path = joinpath(directory,config.output.log_file)
    metadata = run_metadata()
    return with_run_log(log_path; parameters=config, metadata,
                        console=config.output.console, log_every=config.output.log_every) do
        lattice, model, solver = config.lattice, config.model, config.solver
        lattice.boundary == :open || throw(ArgumentError("this example supports open boundaries"))
        solver.method in (:ftlm,:exact) || throw(ArgumentError("method must be :ftlm or :exact"))
        N = lattice.L * lattice.W
        bonds = square_bonds(lattice.L,lattice.W)
        solver.method == :exact && N > solver.exact_max_sites && throw(ArgumentError("exact mode exceeds the site limit"))
        @info "Building model"
        blocks = hubbard_blocks(N,bonds; model...)
        @info "Model ready" sectors=length(blocks) trace_dimension=sum(big(b.dim)*b.multiplicity for b in blocks)
        records = solver.method == :exact ? exact_samples(blocks; maxdim=solver.exact_maxdim, verbose=true) :
            ftlm_samples(blocks; R=solver.R, M=solver.M, seed=solver.seed, sampling=solver.sampling,
                reorthogonalize=solver.reorthogonalize, tol=solver.tol, verbose=true)
        @info "Aggregating thermal results"
        energy = thermal_energy(records,config.beta; nsites=N)
        density = (β=energy.β,values=energy.density,errors=energy.density_errors)
        result = EDResult(parameters=merge(config,(N=N,)),model=merge((name="hubbard",),model),
            lattice=merge(lattice,(bonds=bonds,)),exact=solver.method == :exact ? energy : NamedTuple(),
            observables=(energy=energy,density=density),
            metadata=merge(metadata,(format_version=2,method=string(solver.method),log_path=log_path,
                precision_status=solver.method == :exact ? "dense reference" : "check R and M convergence")))
        log_thermal_results(energy; observables=(density=density,))
        path = save_result(joinpath(directory,config.output.result_file),result)
        @info "Saved" result=path
        return result
    end
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    HubbardExample.main()
end
