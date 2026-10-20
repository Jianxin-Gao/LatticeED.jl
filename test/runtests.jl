using LatticeED
using LinearAlgebra
using Random
using SparseArrays
using FiniteLattices
using Test

# All reference Hamiltonians have at most four sites.
include("reference.jl")
include("solvers.jl")
include("spin.jl")
include("hubbard.jl")
