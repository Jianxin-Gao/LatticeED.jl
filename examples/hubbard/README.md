# Hubbard example

## Model and Hamiltonian

This example computes the thermodynamics of the spinful Hubbard model on an
$L\times W$ square lattice with open boundaries:

$$
H_0 = -t\sum_{\langle i,j\rangle,\sigma}
\left(c^\dagger_{i\sigma}c_{j\sigma}
+c^\dagger_{j\sigma}c_{i\sigma}\right)
+U\sum_i n_{i\uparrow}n_{i\downarrow},
\qquad K=H_0-\mu\hat N.
$$

Each nearest-neighbor bond is counted once, $N=LW$, and $\hat N$ is the total
electron-number operator. The defaults are a 2×2 lattice, $t=1$, $U=8$, and
$\mu=4$ in the grand-canonical ensemble, with weights $e^{-\beta K}$.

The outputs include physical energy $\langle H_0\rangle/N$, grand-canonical
energy $\langle K\rangle/N$, density $\langle\hat N\rangle/N$, and their
sampling errors. A canonical calculation fixes the total electron number Ne
and includes all allowed spin sectors at that particle number.

## Run

From the repository root, prepare the shared example environment once and run:

~~~sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/hubbard/main.jl
~~~

Edit config.jl to change the calculation. The default method is FTLM.
For a small-system exact calculation, set solver.method=:exact in that file,
or use an environment override:

~~~sh
EDJL_HUBBARD_METHOD=exact julia --project=examples examples/hubbard/main.jl
~~~

For a canonical half-filled calculation on the default four-site lattice:

~~~sh
EDJL_HUBBARD_ENSEMBLE=canonical EDJL_HUBBARD_NE=4 \
    julia --project=examples examples/hubbard/main.jl
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

Optional environment overrides use the EDJL_HUBBARD_ prefix with suffixes L, W,
T, U, MU, ENSEMBLE, NE, SYMMETRY, THREADED, METHOD, R, M, SEED, SAMPLING,
REORTHOGONALIZE, TOL, OUTPUT, CONSOLE, and LOG_EVERY. Use true/false for
booleans and omit the colon from symbol values. The remaining settings are
changed directly in config.jl.

| Parameter | Default | Meaning |
|---|---|---|
| `lattice.L` | `2` | number of sites along the first lattice direction. |
| `lattice.W` | `2` | number of sites along the second direction; use 1 for a chain. |
| `lattice.boundary` | `:open` | boundary condition; this example supports open boundaries. |
| `model.t` | `1.0` | real nearest-neighbor hopping amplitude, with the minus sign shown in $H_0$. |
| `model.U` | `8.0` | onsite interaction between up and down electrons. |
| `model.mu` | `4.0` | chemical potential in $K=H_0-\mu\hat N$; at fixed Ne it contributes a constant shift and does not change physical averages. |
| `model.ensemble` | `:grand_canonical` | ensemble, either `:grand_canonical` or `:canonical`. |
| `model.Ne` | `nothing` | fixed total electron number for `:canonical`; it defaults to L*W when the canonical ensemble is selected. Leave it unset for `:grand_canonical`. |
| `model.symmetry` | `:auto` | reduce equivalent sectors using spin exchange and, when applicable, particle-hole symmetry; `:none` keeps all sectors. |
| `model.threaded` | `true` | allow threaded Hamiltonian application for sufficiently large sectors; Julia's thread count is set when starting Julia. |
| `solver.method` | `:ftlm` | calculation method, either `:ftlm` or `:exact`. |
| `solver.R` | `32` | number of random starting states per representative sector in random FTLM. |
| `solver.M` | `64` | requested Lanczos dimension per starting state; small sectors or early convergence can use fewer steps. |
| `solver.seed` | `1234` | seed for reproducible random trace sampling. |
| `solver.sampling` | `:random` | use random states; `:basis` traverses every basis vector for a small-system reference. R is unused in `:basis` mode. |
| `solver.reorthogonalize` | `false` | enable two-pass full reorthogonalization and storage of the Krylov basis. |
| `solver.tol` | `1e-12` | relative Lanczos breakdown tolerance. |
| `solver.exact_max_sites` | `6` | maximum number of sites allowed by this example in exact mode. |
| `solver.exact_maxdim` | `4096` | maximum dimension of any sector in exact mode. |
| `beta` | `0.0:0.05:16.0` | inverse-temperature grid, with $k_{\mathrm B}=1$; beta=0 denotes infinite temperature. |
| `output.root` | `data/` in this example folder | parent directory for unique per-run output folders. |
| `output.result_file` | `result.jld` | result filename within each run folder. |
| `output.log_file` | `run.log` | log filename within each run folder. |
| `output.console` | `true` | also display log events in the terminal. |
| `output.log_every` | `1` | completed-sample logging interval; first and last samples are always logged. |
