# Set W=1 for a chain. Canonical traces use all spin sectors at the chosen Ne.
function example_config(env=ENV)
    parameter(T, name, default) = parse(T, get(env, "EDJL_HUBBARD_" * name, string(default)))
    choice(name, default) = Symbol(get(env, "EDJL_HUBBARD_" * name, string(default)))
    ensemble = choice("ENSEMBLE",:grand_canonical)
    L, W = parameter(Int,"L",2), parameter(Int,"W",2)
    Ne = haskey(env,"EDJL_HUBBARD_NE") ? parameter(Int,"NE",L*W) : ensemble == :canonical ? L*W : nothing
    return (
        lattice=(L=L,W=W,boundary=:open),
        model=(t=parameter(Float64,"T",1.0), U=parameter(Float64,"U",8.0),
            mu=parameter(Float64,"MU",4.0), ensemble=ensemble, Ne=Ne,
            symmetry=choice("SYMMETRY",:auto), threaded=parameter(Bool,"THREADED",true)),
        solver=(method=choice("METHOD",:ftlm), R=parameter(Int,"R",32),
            M=parameter(Int,"M",64), seed=parameter(Int,"SEED",1234),
            sampling=choice("SAMPLING",:random), reorthogonalize=parameter(Bool,"REORTHOGONALIZE",false),
            tol=parameter(Float64,"TOL",1e-12), exact_max_sites=6, exact_maxdim=4096),
        beta=collect(0.0:0.05:16.0),
        output=(root=get(env,"EDJL_HUBBARD_OUTPUT",joinpath(@__DIR__,"data")),
            result_file="result.jld",log_file="run.log",
            console=parameter(Bool,"CONSOLE",true),log_every=parameter(Int,"LOG_EVERY",1)),
    )
end
