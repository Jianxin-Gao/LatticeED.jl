"""
    ThermalBlock(key, dim, build; multiplicity=1, particles=0,
                 physical_shift=0, energy_offset=0, model_id="")

Lazy operator factory for one trace sector. The operator contains an optional
numerical `energy_offset`; reported energies subtract it. `physical_shift` adds
a block-constant observable (e.g. mu*N) to recover physical internal energy.
For symmetry orbits, particles and physical_shift are orbit averages.
"""
struct ThermalBlock{F}
    key::String
    dim::Int
    build::F
    multiplicity::Int
    particles::Float64
    physical_shift::Float64
    energy_offset::Float64
    model_id::String
    members::Vector{Tuple{Int,Int}}
end

function ThermalBlock(key, dim::Int, build; multiplicity::Int=1, particles::Real=0,
                      physical_shift::Real=0, energy_offset::Real=0,
                      model_id="", members=Tuple{Int,Int}[])
    dim > 0 && multiplicity > 0 || throw(ArgumentError("block dimensions and multiplicity must be positive"))
    all(isfinite, (particles, physical_shift, energy_offset)) || throw(ArgumentError("invalid block metadata"))
    return ThermalBlock(string(key), dim, build, multiplicity, Float64(particles),
                        Float64(physical_shift), Float64(energy_offset), string(model_id), collect(members))
end

_digest(x) = bytes2hex(sha256(repr(x)))

"""
    heisenberg_blocks(N, bonds; J=1, hz=0)

Lazy fixed-Sz sectors of J sum_<ij> Si⋅Sj - hz sum_i Szi. All sectors
participate in the trace. Bonds are unique undirected pairs, optionally weighted.
"""
function heisenberg_blocks(N::Int, bonds; J::Real=1, hz::Real=0)
    0 <= N <= 63 || throw(ArgumentError("require 0 <= N <= 63"))
    all(isfinite, (J,hz)) || throw(ArgumentError("model parameters must be finite"))
    bs = _hubbard_bonds(N, bonds)
    model_id = _digest((:heisenberg_v1,N,bs,Float64(J),Float64(hz)))
    exchange = J * Matrix{Float64}(I,3,3)
    fields = [(i,3,-hz) for i in 1:N]
    return [ThermalBlock(string(nup), Int(binomial(big(N),nup)),
        let nup=nup
            () -> SpinHalfOperator(SpinHalfBasis(N,nup), bs; J=exchange, fields)
        end; model_id, members=[(nup,N-nup)]) for nup in 0:N]
end

"""Total Sz (or its equal-weight symmetry-orbit mean) from sector populations."""
function total_Sz(block::ThermalBlock)
    isempty(block.members) && throw(ArgumentError("block has no spin-population metadata"))
    return sum((u-d)/2 for (u,d) in block.members) / length(block.members)
end

_block_hamiltonian(block) = block isa NamedTuple ? block.H : block isa Pair ? block.first : block

function _block_dim(block)
    block isa Pair && return block.second
    block isa NamedTuple || return size(block,1)
    return haskey(block,:dim) ? block.dim : haskey(block,:basis) ? length(block.basis) : size(block.H,1)
end

function _bipartite(N, bonds)
    neighbors = [Int[] for _ in 1:N]
    for (i, j, _) in bonds
        push!(neighbors[i], j); push!(neighbors[j], i)
    end
    colors = zeros(Int, N)
    for root in 1:N
        colors[root] == 0 || continue
        colors[root] = 1
        queue = [root]
        while !isempty(queue)
            i = pop!(queue)
            for j in neighbors[i]
                colors[j] == colors[i] && return false
                if colors[j] == 0
                    colors[j] = -colors[i]
                    push!(queue, j)
                end
            end
        end
    end
    return true
end

"""
    hubbard_blocks(N, bonds; t=1, U=8, mu=4, ensemble=:grand_canonical,
                   Ne=nothing, symmetry=:auto, energy_offset=0, threaded=true)

Generate lazy (Nup,Ndown) sectors. :auto uses spin exchange and, only for a
bipartite graph at mu == U/2, particle–hole symmetry. :none retains all sectors.
Canonical traces include every allowed spin sector at fixed total Ne.
"""
function hubbard_blocks(N::Int, bonds; t::Real=1, U::Real=8, mu::Real=4,
                        ensemble::Symbol=:grand_canonical, Ne=nothing,
                        symmetry::Symbol=:auto, energy_offset::Real=0, threaded::Bool=true)
    _hubbard_dim(N, 0, 0)
    all(isfinite, (t, U, mu, energy_offset)) || throw(ArgumentError("parameters must be finite"))
    ensemble in (:grand_canonical, :canonical) || throw(ArgumentError("unknown ensemble"))
    symmetry in (:auto, :none) || throw(ArgumentError("symmetry must be :auto or :none"))
    if ensemble == :canonical
        Ne isa Integer && 0 <= Ne <= 2N || throw(ArgumentError("canonical ensemble requires 0 <= Ne <= 2N"))
    elseif Ne !== nothing
        throw(ArgumentError("Ne is only meaningful for the canonical ensemble"))
    end
    bs = _hubbard_bonds(N, bonds)
    ph = symmetry == :auto && mu == U / 2 && _bipartite(N, bs) &&
         (ensemble == :grand_canonical || Ne == N)
    keys = [(u, d) for u in 0:N for d in 0:N if ensemble == :grand_canonical || u + d == Ne]
    seen = Set{Tuple{Int,Int}}()
    blocks = ThermalBlock[]
    model_id = _digest((:hubbard_v1, N, bs, Float64(t), Float64(U), Float64(mu), ensemble, Ne))
    for (u, d) in keys
        (u, d) in seen && continue
        orbit = [(u, d)]
        if symmetry == :auto
            push!(orbit, (d, u))
            ph && append!(orbit, [(N-u, N-d), (N-d, N-u)])
        end
        sort!(unique!(orbit))
        union!(seen, orbit)
        particles = sum(a + b for (a, b) in orbit) / length(orbit)
        build = let u=u, d=d, bs=copy(bs)
            () -> HubbardOperator(HubbardBasis(N, u, d), bs; t, U, mu, energy_offset, threaded)
        end
        push!(blocks, ThermalBlock("$(u)_$(d)", _hubbard_dim(N, u, d), build;
              multiplicity=length(orbit), particles, physical_shift=mu * particles,
              energy_offset, model_id, members=orbit))
    end
    return blocks
end

function _as_thermal_blocks(blocks::AbstractVector)
    isempty(blocks) && throw(ArgumentError("blocks must not be empty"))
    return [b isa ThermalBlock ? b : ThermalBlock(string(i), _block_dim(b), () -> _block_hamiltonian(b))
            for (i, b) in enumerate(blocks)]
end
