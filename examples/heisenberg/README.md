# Heisenberg example

## Model and Hamiltonian

This example computes the thermodynamics of the spin-1/2 Heisenberg model on an
$L\times W$ square lattice with open boundaries:

```math
H = J \sum_{\langle i,j\rangle}
\left(S_{i}^{x}\,S_{j}^{x} + S_{i}^{y}\,S_{j}^{y} + S_{i}^{z}\,S_{j}^{z}\right)
- h_{z} \sum_{i=1}^{N} S_{i}^{z}.
```

Each nearest-neighbor bond is counted once, and $N=LW$. The defaults are a 2×2
lattice, $J=1$, and $h_z=0.3$. The thermal trace includes every fixed-$S^z$
sector. The reported observables are the energy $\langle H\rangle/N$ and
magnetization $\langle\sum_i S_i^z\rangle/N$.

## Run

From the repository root, prepare the shared example environment once and run:

~~~sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/heisenberg/main.jl
~~~

Edit config.jl to change the calculation. The default method is FTLM.
For a small-system exact calculation, set solver.method=:exact in that file,
or use an environment override:

~~~sh
EDJL_SPIN_METHOD=exact julia --project=examples examples/heisenberg/main.jl
~~~

Each run creates a new data/run-* directory inside this example folder.
result.jld contains the configuration, observables, and reproducibility metadata;
run.log lists parameters first and then records progress and result summaries.
Output files are ignored by Git.

FTLM error bars estimate trace-sampling uncertainty only. Check convergence in
R and M separately. Exact mode has no sampling uncertainty; a complete :basis
trace still requires sufficient M to eliminate Krylov approximation error.

## Configuration parameters

All parameters are defined in `config.jl`. Solver sampling settings apply to
FTLM; exact-mode size limits apply only to `:exact`.

Optional environment overrides use the EDJL_SPIN_ prefix with suffixes L, W, J,
HZ, METHOD, R, M, SEED, SAMPLING, REORTHOGONALIZE, TOL, OUTPUT, CONSOLE, and
LOG_EVERY. Use true/false for booleans and omit the colon from symbol values.
The remaining settings are changed directly in config.jl.

| Parameter | Default | Meaning |
|---|---|---|
| `lattice.L` | `2` | number of sites along the first lattice direction. |
| `lattice.W` | `2` | number of sites along the second direction; use 1 for a chain. |
| `lattice.boundary` | `:open` | boundary condition; this example supports open boundaries. |
| `model.J` | `1.0` | nearest-neighbor exchange coupling; positive values are antiferromagnetic. |
| `model.hz` | `0.3` | uniform z-directed field, entering the Hamiltonian as $-h_z S_i^z$. |
| `solver.method` | `:ftlm` | calculation method, either `:ftlm` or `:exact`. |
| `solver.R` | `32` | number of random starting states per sector in random FTLM. |
| `solver.M` | `64` | requested Lanczos dimension per starting state; small sectors or early convergence can use fewer steps. |
| `solver.seed` | `1234` | seed for reproducible random trace sampling. |
| `solver.sampling` | `:random` | use random states; `:basis` traverses every basis vector for a small-system reference. R is unused in `:basis` mode. |
| `solver.reorthogonalize` | `false` | enable two-pass full reorthogonalization and storage of the Krylov basis. |
| `solver.tol` | `1e-12` | relative Lanczos breakdown tolerance. |
| `solver.exact_max_sites` | `12` | maximum number of sites allowed by this example in exact mode. |
| `solver.exact_maxdim` | `4096` | maximum dimension of any sector in exact mode. |
| `beta` | `0.0:0.05:16.0` | inverse-temperature grid, with $k_{\mathrm B}=1$; beta=0 denotes infinite temperature. |
| `output.root` | `data/` in this example folder | parent directory for unique per-run output folders. |
| `output.result_file` | `result.jld` | result filename within each run folder. |
| `output.log_file` | `run.log` | log filename within each run folder. |
| `output.console` | `true` | also display log events in the terminal. |
| `output.log_every` | `1` | completed-sample logging interval; first and last samples are always logged. |
