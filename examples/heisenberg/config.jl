# Edit this file to choose the model, solver, temperature grid and output settings.
# Environment overrides are optional; booleans use "true" or "false".
function example_config(env=ENV)
    parameter(T, name, default) = parse(T, get(env, "EDJL_SPIN_" * name, string(default)))
    choice(name, default) = Symbol(get(env, "EDJL_SPIN_" * name, string(default)))
    return (
        lattice=(L=parameter(Int,"L",2), W=parameter(Int,"W",2), boundary=:open),
        model=(J=parameter(Float64,"J",1.0), hz=parameter(Float64,"HZ",0.3)),
        solver=(method=choice("METHOD",:ftlm), R=parameter(Int,"R",32),
            M=parameter(Int,"M",64), seed=parameter(Int,"SEED",1234),
            sampling=choice("SAMPLING",:random), reorthogonalize=parameter(Bool,"REORTHOGONALIZE",false),
            tol=parameter(Float64,"TOL",1e-12), exact_max_sites=12, exact_maxdim=4096),
        beta=collect(0.0:0.05:16.0),
        output=(root=get(env,"EDJL_SPIN_OUTPUT",joinpath(@__DIR__,"data")),
            result_file="result.jld", log_file="run.log",
            console=parameter(Bool,"CONSOLE",true), log_every=parameter(Int,"LOG_EVERY",1)),
    )
end
