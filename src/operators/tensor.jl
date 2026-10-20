using SparseArrays

mutable struct HamiltonianBuilder
    Latt
    Op
    H::AbstractMatrix{ComplexF64}
    use_sparse::Bool
    nsites::Int
    cache::Dict{Any, Any}
end

function HamiltonianBuilder(Latt, Op; use_sparse::Bool=issparse(Op.Id),
                            maxdim::Int=use_sparse ? 1_000_000 : 4096)
    nsites = length(Latt.sites)
    local_dim = size(Op.Id, 1)
    large_dim = big(local_dim)^nsites
    large_dim <= maxdim || throw(ArgumentError("tensor-product dimension $large_dim exceeds maxdim=$maxdim; use a sector matrix-free operator"))
    dim = Int(large_dim)
    H = use_sparse ? spzeros(ComplexF64, dim, dim) : zeros(ComplexF64, dim, dim)
    return HamiltonianBuilder(Latt, Op, H, use_sparse, nsites, Dict{Any, Any}())
end

matrix(builder::HamiltonianBuilder) = builder.H

function _object_cache_id(x)
    return objectid(x)
end

function _cached_multi_site_op!(builder::HamiltonianBuilder, key, ops)
    return get!(builder.cache, key) do
        get_multi_site_op(ops, builder.nsites; use_sparse=builder.use_sparse)
    end
end

function _cached_jw_ops!(builder::HamiltonianBuilder, op::AbstractMatrix, F::AbstractMatrix)
    key = (:jw_ops, _object_cache_id(op), _object_cache_id(F))
    return get!(builder.cache, key) do
        [jw_operator(op, i, builder.nsites, F; use_sparse=builder.use_sparse) for i in 1:builder.nsites]
    end
end

function _bond_sites_and_coeff(bond, coeff)
    if length(bond) == 2
        return bond[1], bond[2], coeff
    elseif length(bond) == 3
        return bond[1], bond[2], coeff * bond[3]
    else
        throw(ArgumentError("bonds must contain (i, j) or (i, j, weight) entries"))
    end
end

"""
    add_onsite!(builder, op; coeff=1, sites=1:builder.nsites)

Add `coeff * op` on each selected site.
"""
function add_onsite!(builder::HamiltonianBuilder, op::AbstractMatrix; coeff=1, sites=1:builder.nsites)
    op_key = _object_cache_id(op)
    for i in sites
        key = (:onsite, i, op_key)
        builder.H += coeff * _cached_multi_site_op!(builder, key, Dict(i => op))
    end
    return builder
end

"""
    add_bond!(builder, op_i, op_j=op_i; coeff=1, bonds=neighbor(builder.Latt), hermitian=false, jw=false, F=nothing)

Add two-site interactions on `bonds`. Each bond may be `(i, j)` or `(i, j, weight)`.
Set `hermitian=true` to also add the Hermitian-conjugate reversed term. Set
`jw=true` and pass `F` to add fermion hopping with Jordan-Wigner strings.
"""
function add_bond!(
    builder::HamiltonianBuilder,
    op_i::AbstractMatrix,
    op_j::AbstractMatrix=op_i;
    coeff=1,
    bonds=neighbor(builder.Latt),
    hermitian::Bool=false,
    jw::Bool=false,
    F::Union{Nothing, AbstractMatrix}=nothing,
)
    if jw
        isnothing(F) && throw(ArgumentError("F must be provided when jw=true"))
        C_i = _cached_jw_ops!(builder, op_i, F)
        C_j = op_j === op_i ? C_i : _cached_jw_ops!(builder, op_j, F)
        for bond in bonds
            i, j, c = _bond_sites_and_coeff(bond, coeff)
            builder.H += c * (C_i[i]' * C_j[j])
            builder.H += conj(c) * (C_j[j]' * C_i[i])
        end
        return builder
    end

    op_i_key = _object_cache_id(op_i)
    op_j_key = _object_cache_id(op_j)
    for bond in bonds
        i, j, c = _bond_sites_and_coeff(bond, coeff)
        key = (:bond, i, j, op_i_key, op_j_key)
        builder.H += c * _cached_multi_site_op!(builder, key, Dict(i => op_i, j => op_j))
        if hermitian
            hc_key = (:bond_hc, i, j, op_i_key, op_j_key)
            builder.H += conj(c) * _cached_multi_site_op!(builder, hc_key, Dict(i => op_i', j => op_j'))
        end
    end
    return builder
end

function hubbard_hamiltonian(Latt, t, U, μ, Op)
    builder = HamiltonianBuilder(Latt, Op)

    add_bond!(builder, Op.a_up; coeff=-t, F=Op.F, jw=true)
    add_bond!(builder, Op.a_down; coeff=-t, F=Op.F, jw=true)
    add_onsite!(builder, Op.n_up * Op.n_down; coeff=U)
    add_onsite!(builder, Op.n_up + Op.n_down; coeff=-μ)

    return matrix(builder)
end

function heisenberg_hamiltonian(Latt, J, Op)
    builder = HamiltonianBuilder(Latt, Op)

    bonds = neighbor(Latt)
    for component in (Op.Sx, Op.Sy, Op.Sz)
        add_bond!(builder, component; coeff=J, bonds)
    end

    return matrix(builder)
end
