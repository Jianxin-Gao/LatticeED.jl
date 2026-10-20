"""
    spin_operators(d::Int=2; use_sparse::Bool=false)

Construct spin operators for local Hilbert-space dimension `d`, i.e. spin
`S = (d - 1) / 2`, in the basis `m = S, S-1, ..., -S`.
"""
function spin_operators(d::Int=2; use_sparse::Bool=false)
    d < 1 && throw(ArgumentError("local Hilbert-space dimension d must be positive"))

    M(x) = use_sparse ? sparse(x) : Matrix(x)

    S = (d - 1) / 2
    mvals = [S - (i - 1) for i in 1:d]
    states = [reshape([i == j ? 1.0 : 0.0 for i in 1:d], d, 1) for j in 1:d]

    S₊ = zeros(Float64, d, d) # S₊ |m> = sqrt(S(S+1)-m(m+1)) |m+1>
    for col in 2:d
        m = mvals[col]
        S₊[col - 1, col] = sqrt(S * (S + 1) - m * (m + 1))
    end
    S₋ = S₊' # S₋ is the lowering operator
    Sz = diagm(0 => mvals) # Sz is the z-component of the spin operator
    Sx = (S₊ + S₋) / 2 # Sx is the x-component of the spin operator
    Sy = (S₊ - S₋) / (2im) # Sy
    S² = S * (S + 1) * Matrix{Float64}(I, d, d)

    Id = Matrix(I, d, d) # Id is the identity operator

    return (
        dim = d,
        spin = S,
        states = map(M, states),
        mvals = mvals,
        u   = M(states[1]),
        d   = M(states[end]),
        S₊   = M(S₊),
        S₋   = M(S₋),
        Sz   = M(Sz),
        Sx   = M(Sx),
        Sy   = M(Sy),
        S²   = M(S²),
        Id   = M(Id)
    )
end

function fermion_operators(; use_sparse::Bool=false)
    M(x) = use_sparse ? sparse(x) : Matrix(x)

    ud = reshape([1,0,0,0], 4, 1) # ud is the two occupy state
    u = reshape([0,1,0,0], 4, 1) # u is the up spin state
    d = reshape([0,0,1,0], 4, 1) # d is the down spin state
    vac = reshape([0,0,0,1], 4, 1) # vac is the vacuum state

    a_down = [
        0 0 0 0;
        -1 0 0 0;
        0 0 0 0;
        0 0 1 0
    ] # a_down is the annihilation operator for a down spin fermion

    a_up = [
        0 0 0 0;
        0 0 0 0;
        1 0 0 0;
        0 1 0 0
    ] # a_up is the annihilation operator for an up spin fermion

    F = diagm(0 => [1, -1, -1, 1]) # F is the JW string operator
    Id = Matrix(I, 4, 4) # Id is the identity operator
    n_down = a_down' * a_down # n_down is the number operator for down spin fermions
    n_up = a_up' * a_up # n_up is the number operator for up spin fermions
    S₊ = a_up' * a_down # S₊ is the raising operator for spin
    S₋ = a_down' * a_up # S₋ is the lowering operator
    Sz = (n_up - n_down) / 2 # Sz is the z-component of the spin operator
    Sx = (S₊ + S₋) / 2 # Sx is the x-component of the spin operator
    Sy = (S₊ - S₋) / (2im) # Sy
    S² = Sx^2 + Sy^2 + Sz^2 # S² operator

    return (
        vac  = M(vac),
        u   = M(u),
        d   = M(d),
        ud   = M(ud),
        a_up = M(a_up),
        a_down = M(a_down),
        Id   = M(Id),
        F    = M(F),
        n_up = M(n_up),
        n_down = M(n_down),
        Sz   = M(Sz),
        Sx   = M(Sx),
        Sy   = M(Sy),
        S²   = M(S²),
        S₊   = M(S₊),
        S₋   = M(S₋)
    )
end

"""
    get_multi_site_op(ops_dict::AbstractDict{Int, <:AbstractMatrix}, L::Int; use_sparse::Bool=false)

Construct a multi-site operator for a system of length `L`.

- `ops_dict` maps site indices (1-based) to local operators.
- Other sites are filled with identity.
- `use_sparse=true` returns a sparse matrix, otherwise dense (defult).
"""
function get_multi_site_op(ops_dict::AbstractDict{Int, <:AbstractMatrix}, L::Int; use_sparse::Bool=false)
    T = promote_type(map(eltype, values(ops_dict))...)
    ops = Vector{AbstractMatrix{T}}(undef, L)
    d = size(first(values(ops_dict)))[1] # dimension of local operators
    identity = use_sparse ? spdiagm(0 => ones(T, d)) : Matrix{T}(I, d, d)
    fill!(ops, identity)

    for (site, op) in ops_dict
        ops[site] = use_sparse ? sparse(T.(op)) : Matrix{T}(op)
    end

    return reduce(kron, ops)
end

"""
    jw_operator(operator::AbstractMatrix{T}, site::Int, L::Int, F::AbstractMatrix{T}; sparse::Bool=false) where T

Construct a Jordan-Wigner string operator at a given site.

- Sites before `site` get the F operator.
- The site itself gets `operator`.
- Remaining sites get identity.
- `sparse=true` returns sparse matrix.
"""
function jw_operator(operator::AbstractMatrix{T}, site::Int, L::Int, F::AbstractMatrix{T}; use_sparse::Bool=false) where T
    I4 = use_sparse ? spdiagm(0 => ones(T, 4)) : Matrix(I, 4, 4)

    ops = Vector{AbstractMatrix{T}}(undef, L)
    for i in 1:L
        if i < site
            ops[i] = F
        elseif i == site
            ops[i] = operator
        else
            ops[i] = I4
        end
    end
    return reduce(kron, ops)
end

function pairing_operator(i::Int, j::Int, L::Int, a_up::AbstractMatrix, a_down::AbstractMatrix, F::AbstractMatrix; use_sparse::Bool=false)
    a_up_i = jw_operator(a_up, i, L, F; use_sparse)
    a_down_i = jw_operator(a_down, i, L, F; use_sparse)
    a_up_j = jw_operator(a_up, j, L, F; use_sparse)
    a_down_j = jw_operator(a_down, j, L, F; use_sparse)
    return (a_up_i * a_down_j - a_down_i * a_up_j) / sqrt(2)
end

function chiral_op(i::Int, j::Int, k::Int, Sx::AbstractMatrix, Sy::AbstractMatrix, Sz::AbstractMatrix, L::Int; use_sparse = false)
    Six = get_multi_site_op(Dict(i => Sx), L; use_sparse)
    Siy = get_multi_site_op(Dict(i => Sy), L; use_sparse)
    Siz = get_multi_site_op(Dict(i => Sz), L; use_sparse)

    Sjx = get_multi_site_op(Dict(j => Sx), L; use_sparse)
    Sjy = get_multi_site_op(Dict(j => Sy), L; use_sparse)
    Sjz = get_multi_site_op(Dict(j => Sz), L; use_sparse)

    Skx = get_multi_site_op(Dict(k => Sx), L; use_sparse)
    Sky = get_multi_site_op(Dict(k => Sy), L; use_sparse)
    Skz = get_multi_site_op(Dict(k => Sz), L; use_sparse)

    return Six * (Sjy * Skz - Sjz * Sky) + Siy * (Sjz * Skx - Sjx * Skz) + Siz * (Sjx * Sky - Sjy * Skx)
end
