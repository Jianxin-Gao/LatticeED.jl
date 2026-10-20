"""Unique undirected nearest-neighbour bonds on an open L × W square (x varies fastest)."""
function square_bonds(L::Int, W::Int)
    L > 0 && W > 0 || throw(ArgumentError("lattice sizes must be positive"))
    bonds = Tuple{Int,Int}[]
    for y in 1:W, x in 1:L
        i = x + (y - 1) * L
        x < L && push!(bonds, (i, i + 1))
        y < W && push!(bonds, (i, i + L))
    end
    return bonds
end

function _hubbard_bonds(N, bonds)
    result = Tuple{Int,Int,Float64}[]
    for bond in bonds
        length(bond) in (2, 3) || throw(ArgumentError("bonds require (i,j[,weight])"))
        i, j = Int(bond[1]), Int(bond[2])
        1 <= i <= N && 1 <= j <= N && i != j || throw(ArgumentError("invalid bond sites"))
        w = length(bond) == 3 ? Float64(bond[3]) : 1.0
        isfinite(w) || throw(ArgumentError("bond weights must be finite and real"))
        iszero(w) || push!(result, (min(i, j), max(i, j), w))
    end
    return sort!(result)
end

struct _HoppingRows
    ptr::Vector{Int}
    targets::Vector{Int}
    values::Vector{Float64}
end

function _hopping_rows(states, index, bonds, t)
    ptr, targets, values = Int[1], Int[], Float64[]
    for s in states
        for (i, j, weight) in bonds
            a, b = UInt64(1) << (i - 1), UInt64(1) << (j - 1)
            ((s & a) == 0) == ((s & b) == 0) && continue
            mask = ((UInt64(1) << (j - 1)) - 1) ⊻ ((UInt64(1) << i) - 1)
            sign = isodd(count_ones(s & mask)) ? -1.0 : 1.0
            push!(targets, index[s ⊻ a ⊻ b])
            push!(values, -t * weight * sign)
        end
        push!(ptr, length(targets) + 1)
    end
    return _HoppingRows(ptr, targets, values)
end

"""
    HubbardOperator(basis, bonds; t=1, U=8, mu=4, energy_offset=0, threaded=true)

Real matrix-free K = hopping + U*double_occupancy - mu*N + energy_offset*I.
Only single-spin hopping maps and the product-space diagonal are stored.
An offset is a numerical shift, not a change to the physical model.
"""
struct HubbardOperator <: AbstractMatrix{Float64}
    basis::HubbardBasis
    up_hops::_HoppingRows
    down_hops::_HoppingRows
    diagonal::Vector{Float64}
    threaded::Bool
end

function HubbardOperator(b::HubbardBasis, bonds; t::Real=1, U::Real=8, mu::Real=4,
                         energy_offset::Real=0, threaded::Bool=true)
    all(isfinite, (t, U, mu, energy_offset)) || throw(ArgumentError("parameters must be finite"))
    bs = _hubbard_bonds(b.N, bonds)
    up = _hopping_rows(b.up, b.up_index, bs, Float64(t))
    down = b.Nup == b.Ndown ? up : _hopping_rows(b.down, b.down_index, bs, Float64(t))
    diagonal = Vector{Float64}(undef, length(b))
    nu = length(b.up)
    Threads.@threads for d in eachindex(b.down)
        @inbounds for u in eachindex(b.up)
            diagonal[u + (d - 1) * nu] = U * count_ones(b.up[u] & b.down[d]) -
                                        mu * (b.Nup + b.Ndown) + energy_offset
        end
    end
    return HubbardOperator(b, up, down, diagonal, threaded)
end

Base.size(H::HubbardOperator) = (length(H.basis), length(H.basis))
Base.size(H::HubbardOperator, d::Integer) = d <= 0 ? throw(ArgumentError("dimension must be positive")) : d <= 2 ? length(H.basis) : 1
LinearAlgebra.ishermitian(::HubbardOperator) = true
LinearAlgebra.issymmetric(::HubbardOperator) = true

function _hubbard_rows!(y, H, x, first, last)
    nu = length(H.basis.up)
    up, down = H.up_hops, H.down_hops
    @inbounds for row in first:last
        d0, u0 = divrem(row - 1, nu)
        u, d = u0 + 1, d0 + 1
        value = H.diagonal[row] * x[row]
        for k in up.ptr[u]:(up.ptr[u + 1] - 1)
            value += up.values[k] * x[up.targets[k] + d0 * nu]
        end
        for k in down.ptr[d]:(down.ptr[d + 1] - 1)
            value += down.values[k] * x[u + (down.targets[k] - 1) * nu]
        end
        y[row] = value
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, H::HubbardOperator, x::AbstractVector)
    n = size(H, 1)
    length(y) == length(x) == n || throw(DimensionMismatch("Hubbard matvec dimensions"))
    Base.mightalias(x, y) && throw(ArgumentError("mul! input and output must not alias"))
    if H.threaded && Threads.nthreads() > 1 && n >= 4096
        Threads.@threads for chunk in 1:cld(n, 4096)
            _hubbard_rows!(y, H, x, 4096 * (chunk - 1) + 1, min(4096 * chunk, n))
        end
    else
        _hubbard_rows!(y, H, x, 1, n)
    end
    return y
end
Base.:*(H::HubbardOperator, x::AbstractVector) = mul!(similar(x, promote_type(Float64, eltype(x))), H, x)

function Base.getindex(H::HubbardOperator, i::Int, j::Int)
    checkbounds(H, i, j)
    nu = length(H.basis.up)
    di, ui = divrem(i - 1, nu)
    dj, uj = divrem(j - 1, nu)
    value = i == j ? H.diagonal[i] : 0.0
    if di == dj
        for k in H.up_hops.ptr[ui + 1]:(H.up_hops.ptr[ui + 2] - 1)
            H.up_hops.targets[k] == uj + 1 && (value += H.up_hops.values[k])
        end
    end
    if ui == uj
        for k in H.down_hops.ptr[di + 1]:(H.down_hops.ptr[di + 2] - 1)
            H.down_hops.targets[k] == dj + 1 && (value += H.down_hops.values[k])
        end
    end
    return value
end

function Base.Matrix(H::HubbardOperator)
    size(H, 1) <= 4096 || throw(ArgumentError("dense Hubbard conversion limited to dimension 4096"))
    result = zeros(size(H))
    nu = length(H.basis.up)
    for row in axes(H,1)
        result[row,row] = H.diagonal[row]
        d0, u0 = divrem(row-1, nu)
        for k in H.up_hops.ptr[u0+1]:(H.up_hops.ptr[u0+2]-1)
            result[row,H.up_hops.targets[k]+d0*nu] += H.up_hops.values[k]
        end
        for k in H.down_hops.ptr[d0+1]:(H.down_hops.ptr[d0+2]-1)
            result[row,u0+1+(H.down_hops.targets[k]-1)*nu] += H.down_hops.values[k]
        end
    end
    return result
end
