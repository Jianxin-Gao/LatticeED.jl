struct SpinHalfBasis
    N::Int
    Nup::Union{Nothing, Int}
    states::Vector{UInt64}
    index::Dict{UInt64, Int}
end

function SpinHalfBasis(N::Int)
    N < 0 && throw(ArgumentError("N must be non-negative"))
    N <= 64 || throw(ArgumentError("SpinHalfBasis uses UInt64 states and requires N <= 64"))

    if N == 64
        throw(ArgumentError("full SpinHalfBasis(N) requires N < 64 to enumerate 2^N states"))
    end

    states = collect(UInt64(0):(UInt64(1) << N) - UInt64(1))
    index = Dict(state => i for (i, state) in enumerate(states))
    return SpinHalfBasis(N, nothing, states, index)
end

function SpinHalfBasis(N::Int, Nup::Int)
    N < 0 && throw(ArgumentError("N must be non-negative"))
    0 <= Nup <= N || throw(ArgumentError("Nup must satisfy 0 <= Nup <= N"))
    N <= 64 || throw(ArgumentError("SpinHalfBasis uses UInt64 states and requires N <= 64"))

    states = UInt64[]
    function build(site::Int, nleft::Int, state::UInt64)
        if nleft == 0
            push!(states, state)
            return
        end
        site > N && return

        remaining = N - site
        if remaining >= nleft
            build(site + 1, nleft, state)
        end
        build(site + 1, nleft - 1, state | (UInt64(1) << (site - 1)))
    end
    build(1, Nup, UInt64(0))

    index = Dict(state => i for (i, state) in enumerate(states))
    return SpinHalfBasis(N, Nup, states, index)
end

Base.length(basis::SpinHalfBasis) = length(basis.states)
Base.getindex(basis::SpinHalfBasis, i::Int) = basis.states[i]

state_index(basis::SpinHalfBasis, state::UInt64) = basis.index[state]
is_fixed_sz(basis::SpinHalfBasis) = !isnothing(basis.Nup)
function total_Sz(basis::SpinHalfBasis)
    is_fixed_sz(basis) || throw(ArgumentError("total_Sz is only defined for fixed-Sz SpinHalfBasis(N, Nup)"))
    return basis.Nup - basis.N / 2
end

mutable struct SpinHalfBuilder
    basis::SpinHalfBasis
    H::SparseMatrixCSC{ComplexF64, Int}
end

function SpinHalfBuilder(basis::SpinHalfBasis)
    dim = length(basis)
    return SpinHalfBuilder(basis, spzeros(ComplexF64, dim, dim))
end

matrix(builder::SpinHalfBuilder) = builder.H
