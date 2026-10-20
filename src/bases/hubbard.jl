"""
    HubbardBasis(N, Nup, Ndown)

Product of two fixed-population UInt64 bases (site 1 is the low bit).
Up-spin index varies fastest. Creation operators are ordered as all up sites,
then all down sites, in ascending site order. No product-state table is stored.
"""
struct HubbardBasis
    N::Int
    Nup::Int
    Ndown::Int
    up::Vector{UInt64}
    down::Vector{UInt64}
    up_index::Dict{UInt64,Int}
    down_index::Dict{UInt64,Int}
    dim::Int
end

function _hubbard_dim(N, nu, nd)
    0 <= N <= 63 || throw(ArgumentError("HubbardBasis requires 0 <= N <= 63"))
    0 <= nu <= N && 0 <= nd <= N || throw(ArgumentError("invalid particle numbers"))
    d = binomial(big(N), nu) * binomial(big(N), nd)
    d <= typemax(Int) || throw(ArgumentError("sector dimension exceeds Int capacity"))
    return Int(d)
end

function _occupation_states(N, n)
    states = UInt64[]
    function visit(first, left, state)
        if left == 0
            push!(states, state)
        else
            for site in first:(N - left + 1)
                visit(site + 1, left - 1, state | (UInt64(1) << (site - 1)))
            end
        end
    end
    visit(1, n, UInt64(0))
    sort!(states)
    return states
end

function HubbardBasis(N::Int, nu::Int, nd::Int)
    dim = _hubbard_dim(N, nu, nd)
    up = _occupation_states(N, nu)
    down = nu == nd ? up : _occupation_states(N, nd)
    ui = Dict(s => i for (i, s) in enumerate(up))
    di = nu == nd ? ui : Dict(s => i for (i, s) in enumerate(down))
    return HubbardBasis(N, nu, nd, up, down, ui, di, dim)
end

Base.length(b::HubbardBasis) = b.dim
function Base.getindex(b::HubbardBasis, i::Int)
    1 <= i <= length(b) || throw(BoundsError(b, i))
    d, u = divrem(i - 1, length(b.up))
    return (b.up[u + 1], b.down[d + 1])
end
state_index(b::HubbardBasis, state::Tuple{UInt64,UInt64}) =
    b.up_index[state[1]] + (b.down_index[state[2]] - 1) * length(b.up)

"""Return tensor-basis indices and phases: |product i⟩ = phase[i] |tensor index[i]⟩."""
function tensor_basis_map(b::HubbardBasis; maxdim=4096)
    length(b) <= maxdim || throw(ArgumentError("basis map is a small-system reference operation"))
    b.N <= 30 || throw(ArgumentError("tensor indices exceed the reference size limit"))
    indices = Int[]
    phases = Int[]
    for k in 1:length(b)
        up, down = b[k]
        index, inversions = 1, 0
        for site in 1:b.N
            u = Int((up >> (site - 1)) & 1)
            d = Int((down >> (site - 1)) & 1)
            index += (3 - 2u - d) * 4^(b.N - site)
            inversions += d * count_ones(up >> site)
        end
        push!(indices, index)
        push!(phases, isodd(inversions) ? -1 : 1)
    end
    return indices, phases
end
