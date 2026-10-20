# LatticeED.jl

Exact diagonalization and finite-temperature Lanczos methods for quantum lattice models in Julia.

## Introduction

The finite-temperature Lanczos method (FTLM) estimates thermal averages without
computing the full many-body spectrum. It combines random sampling of the
Hilbert-space trace with Lanczos diagonalization in small Krylov subspaces, so
thermal quantities can be obtained from repeated Hamiltonian-vector products.
See [Jaklič and Prelovšek, Phys. Rev. B 49, 5065 (1994)](https://doi.org/10.1103/PhysRevB.49.5065)
for the original method.

For inverse temperature $\beta=1/T$ with $k_{\mathrm B}=1$, the equilibrium average is

$$
\langle O\rangle_\beta =
\frac{\mathrm{Tr}\left(Oe^{-\beta H}\right)}
     {\mathrm{Tr}\left(e^{-\beta H}\right)}.
$$

The number of random states controls trace-sampling uncertainty, while the
Lanczos dimension controls the Krylov approximation. Convergence in these two
parameters should be checked separately.

## Installation

Requires Julia 1.11 or later within the Julia 1.x series.

~~~julia
using Pkg
Pkg.add(url="https://github.com/Jianxin-Gao/LatticeED.jl")
~~~

To work from a local checkout:

~~~sh
git clone https://github.com/Jianxin-Gao/LatticeED.jl
cd LatticeED.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
~~~

## A Quick Start

Calculate the thermal energy of a spinful Hubbard model on a 2×2 square lattice
with open boundaries, hopping $t=1$, interaction $U=8$, and chemical potential
$\mu=4$. The grand-canonical trace includes all particle-number sectors.

Run the following in Julia after installation, or start Julia with
`julia --project=.` from the repository root:

~~~julia
using LatticeED

L, W = 2, 2
N = L * W
bonds = square_bonds(L, W)  # Four unique nearest-neighbor bonds, open boundaries.
blocks = hubbard_blocks(N, bonds; t=1.0, U=8.0, mu=4.0)

beta = [0.0, 0.5, 1.0, 2.0]
energy = ftlm_energy(blocks, beta; R=8, M=36, seed=1234, nsites=N)

for i in eachindex(beta)
    println((
        beta=beta[i],
        energy_per_site=energy.energy_per_site[i],
        sampling_error=energy.energy_errors[i],
        density=energy.density[i],
    ))
end
~~~

Here `energy_per_site` is the physical internal energy $\langle H_0\rangle/N$.
`grand_energy_per_site` instead gives $\langle H_0-\mu\hat N\rangle/N$.
The reported error estimates describe random trace sampling and exclude
finite-Krylov-dimension bias. Use `logZ` for the logarithm of the partition function.

For complete configuration and run instructions, see the
[Heisenberg example](examples/heisenberg/README.md) and
[Hubbard example](examples/hubbard/README.md).

Licensed under the [MIT License](LICENSE).
