module LatticeED
using LinearAlgebra, SparseArrays, Random
using FiniteLattices
import JLD2
using SHA, Dates, Logging, Printf
import Pkg
export get_multi_site_op, thermal_expectation, pairing_operator, chiral_op
export SpinHalfBasis, SpinHalfBuilder, SpinHalfOperator, SpinHalfTerm, SpinHalfTermOperator, add_spin_bond!, add_spin_field!, add_spin_term!, state_index, total_Sz, is_fixed_sz
export lanczos, ftlm_energy, ftlm_expectation
export EDResult, save_result, load_result
export HamiltonianBuilder, add_onsite!, add_bond!, matrix

export spin_operators, fermion_operators, jw_operator, hubbard_hamiltonian, heisenberg_hamiltonian
export HubbardBasis, HubbardOperator, hubbard_blocks, heisenberg_blocks, square_bonds, tensor_basis_map
export ThermalBlock, LanczosSample, ftlm_samples, exact_samples, thermal_energy, exact_energy
export thermal_block_observable
export sample_seed, save_sample, load_samples, run_ftlm!, source_fingerprint
export compare_convergence, with_run_log, run_metadata, log_thermal_results

include("operators/local.jl")
include("bases/spin.jl")
include("operators/tensor.jl")
include("operators/spin.jl")
include("bases/hubbard.jl")
include("operators/hubbard.jl")
include("solvers/lanczos.jl")
include("thermodynamics/blocks.jl")
include("thermodynamics/legacy_exact.jl")
include("io/logging.jl")
include("solvers/sampling.jl")
include("thermodynamics/energy.jl")
include("thermodynamics/experimental.jl")
include("io/results.jl")
include("io/checkpoints.jl")
include("thermodynamics/convergence.jl")

end # module
