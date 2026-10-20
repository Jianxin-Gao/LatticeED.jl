_iszero_approx(x; atol=1e-12) = isapprox(x, zero(x); atol)

function _spinhalf_exchange_matrix(J)
    result = Matrix{ComplexF64}(J)
    size(result) == (3,3) || throw(ArgumentError("J must be 3x3 in x,y,z order"))
    all(isfinite, result) || throw(ArgumentError("exchange coefficients must be finite"))
    return result
end

function _conserves_total_sz(J; atol=1e-12)
    return all(_iszero_approx(J[i,j]; atol) for (i,j) in ((1,3),(2,3),(3,1),(3,2))) &&
        isapprox(J[1,1], J[2,2]; atol) && isapprox(J[1,2], -J[2,1]; atol)
end

function _check_spinhalf_sector(basis, sector)
    sector in (:auto, :fixed_sz, :full) || throw(ArgumentError("unknown spin sector"))
    checked = sector == :auto ? (is_fixed_sz(basis) ? :fixed_sz : :full) : sector
    (checked == :fixed_sz) == is_fixed_sz(basis) || throw(ArgumentError("sector does not match the basis"))
    return checked
end

"""
    SpinHalfTerm(sites, axes, coeff=1)

A product of spin-1/2 operators on distinct sites. Axes 1,2,3 mean x,y,z.
Fixed-Sz term operators represent projected observables: out-of-sector actions
are dropped. Hamiltonian exchange/field constructors instead enforce conservation.
"""
struct SpinHalfTerm
    sites::Vector{Int}
    axes::Vector{Int}
    coeff::ComplexF64
    function SpinHalfTerm(sites, axes, coeff=1)
        s, a = Int.(collect(sites)), Int.(collect(axes))
        !isempty(s) && length(s) == length(a) || throw(ArgumentError("sites and axes must have equal nonzero length"))
        length(unique(s)) == length(s) || throw(ArgumentError("repeated sites are not supported"))
        all(x -> 1 <= x <= 3, a) || throw(ArgumentError("spin axes must be 1,2,3"))
        isfinite(coeff) || throw(ArgumentError("term coefficients must be finite"))
        new(s, a, ComplexF64(coeff))
    end
end

_integer_collection(x) = (x isa AbstractVector || x isa Tuple) && all(v -> v isa Integer, x)
_raw_spin_term(x) = x isa Tuple && length(x) in (2,3) && _integer_collection(x[1]) && _integer_collection(x[2])

function _normalize_spinhalf_terms(basis, terms)
    input = terms isa SpinHalfTerm || _raw_spin_term(terms) ? (terms,) : terms
    result = SpinHalfTerm[]
    for raw in input
        term = raw isa SpinHalfTerm ? raw : _raw_spin_term(raw) ? SpinHalfTerm(raw...) :
            throw(ArgumentError("expected SpinHalfTerm or (sites, axes[, coeff])"))
        all(i -> 1 <= i <= basis.N, term.sites) || throw(ArgumentError("spin site outside the lattice"))
        _iszero_approx(term.coeff) || push!(result, term)
    end
    return result
end

# A product acts by one XOR and one parity calculation, regardless of its order.
struct _CompiledSpinTerm{T}
    flip::UInt64
    sign::UInt64
    coeff::T
end

function _compile_spin_terms(terms)
    combined = Dict{Tuple{UInt64,UInt64},ComplexF64}()
    for term in terms
        flip, sign, coeff = UInt64(0), UInt64(0), term.coeff
        for (site, axis) in zip(term.sites, term.axes)
            bit = UInt64(1) << (site-1)
            axis != 3 && (flip |= bit)
            axis != 1 && (sign |= bit)
            coeff *= axis == 1 ? 0.5 : axis == 2 ? -0.5im : -0.5
        end
        key = (flip, sign)
        combined[key] = get(combined, key, 0.0im) + coeff
    end
    entries = sort!([(key, value) for (key,value) in combined if !iszero(value)]; by=first)
    T = all(isreal(last(entry)) for entry in entries) ? Float64 : ComplexF64
    return _CompiledSpinTerm{T}[_CompiledSpinTerm{T}(key[1], key[2], T == Float64 ? real(value) : value) for (key,value) in entries]
end

abstract type _SpinOperator{T} <: AbstractMatrix{T} end

struct SpinHalfTermOperator{T} <: _SpinOperator{T}
    basis::SpinHalfBasis
    terms::Vector{SpinHalfTerm}
    compiled::Vector{_CompiledSpinTerm{T}}
    sector::Symbol
end

"""Matrix-free projected multi-site spin products; never stores a Hilbert-space matrix."""
function SpinHalfTermOperator(basis::SpinHalfBasis, terms=(); sector::Symbol=:auto)
    checked = _check_spinhalf_sector(basis, sector)
    normalized = _normalize_spinhalf_terms(basis, terms)
    return SpinHalfTermOperator(basis, normalized, _compile_spin_terms(normalized), checked)
end

struct SpinHalfOperator{T} <: _SpinOperator{T}
    basis::SpinHalfBasis
    compiled::Vector{_CompiledSpinTerm{T}}
    sector::Symbol
end

"""
    SpinHalfOperator(basis, bonds=(); J=zeros(3,3), fields=(), sector=:auto)

Matrix-free exchange and fields. A field is (site, [hx,hy,hz]) or
(site, axis, coefficient). Fixed-Sz Hamiltonians must conserve total Sz.
Real matrix elements use real storage; complex interactions remain supported.
"""
function SpinHalfOperator(basis::SpinHalfBasis, bonds=(); J=zeros(3,3), fields=(), sector::Symbol=:auto)
    checked = _check_spinhalf_sector(basis, sector)
    exchange = _spinhalf_exchange_matrix(J)
    checked == :fixed_sz && !_conserves_total_sz(exchange) &&
        throw(ArgumentError("exchange does not conserve Sz; use SpinHalfBasis(N)"))
    terms = SpinHalfTerm[]
    for bond in bonds
        i, j, weight = _bond_sites_and_coeff(bond, 1)
        1 <= i <= basis.N && 1 <= j <= basis.N && i != j || throw(ArgumentError("invalid spin bond"))
        isfinite(weight) || throw(ArgumentError("bond weights must be finite"))
        for a in 1:3, b in 1:3
            coeff = weight * exchange[a,b]
            _iszero_approx(coeff) || push!(terms, SpinHalfTerm((i,j), (a,b), coeff))
        end
    end
    for field in fields
        if length(field) == 2
            site, h = field
            length(h) == 3 || throw(ArgumentError("fields require three components"))
            components = enumerate(h)
        elseif length(field) == 3
            site, axis, coeff = field
            components = ((axis, coeff),)
        else
            throw(ArgumentError("expected (site,h) or (site,axis,coefficient)"))
        end
        1 <= site <= basis.N || throw(ArgumentError("field site outside the lattice"))
        for (axis, coeff) in components
            1 <= axis <= 3 && isfinite(coeff) || throw(ArgumentError("invalid field component"))
            _iszero_approx(coeff) && continue
            checked == :fixed_sz && axis != 3 && throw(ArgumentError("field does not conserve Sz; use SpinHalfBasis(N)"))
            push!(terms, SpinHalfTerm((site,), (axis,), coeff))
        end
    end
    return SpinHalfOperator(basis, _compile_spin_terms(terms), checked)
end

Base.size(op::_SpinOperator) = (length(op.basis), length(op.basis))
Base.size(op::_SpinOperator, d::Integer) = d > 0 ? (d <= 2 ? length(op.basis) : 1) : throw(ArgumentError("dimension must be positive"))
_spin_row(basis, state) = is_fixed_sz(basis) ? get(basis.index, state, 0) : Int(state) + 1
_spin_coefficient(term, state) = isodd(count_ones(state & term.sign)) ? -term.coeff : term.coeff

function LinearAlgebra.mul!(y::AbstractVector, op::_SpinOperator, x::AbstractVector)
    length(y) == length(x) == length(op.basis) || throw(DimensionMismatch("spin matvec dimensions"))
    Base.mightalias(x, y) && throw(ArgumentError("mul! input and output must not alias"))
    fill!(y, zero(eltype(y)))
    @inbounds for (col, state) in enumerate(op.basis.states)
        iszero(x[col]) && continue
        for term in op.compiled
            row = _spin_row(op.basis, state ⊻ term.flip)
            row == 0 || (y[row] += _spin_coefficient(term, state) * x[col])
        end
    end
    return y
end

Base.:*(op::_SpinOperator, x::AbstractVector) = mul!(similar(x, promote_type(eltype(op), eltype(x))), op, x)

function Base.getindex(op::_SpinOperator, row::Int, col::Int)
    checkbounds(op, row, col)
    state, target = op.basis.states[col], op.basis.states[row]
    result = zero(eltype(op))
    for term in op.compiled
        state ⊻ term.flip == target && (result += _spin_coefficient(term, state))
    end
    return result
end

function SparseArrays.sparse(op::_SpinOperator)
    rows, cols, values = Int[], Int[], eltype(op)[]
    for (col,state) in enumerate(op.basis.states), term in op.compiled
        row = _spin_row(op.basis, state ⊻ term.flip)
        row == 0 && continue
        push!(rows, row); push!(cols, col); push!(values, _spin_coefficient(term, state))
    end
    return sparse(rows, cols, values, size(op)...)
end

function Base.Matrix(op::_SpinOperator)
    size(op,1) <= 4096 || throw(ArgumentError("dense spin conversion limited to dimension 4096"))
    result = zeros(eltype(op), size(op))
    for (col,state) in enumerate(op.basis.states), term in op.compiled
        row = _spin_row(op.basis, state ⊻ term.flip)
        row == 0 || (result[row,col] += _spin_coefficient(term, state))
    end
    return result
end

"""Add exchange through the same normalized terms as matrix-free application."""
function add_spin_bond!(builder::SpinHalfBuilder, bonds; J, sector::Symbol=:auto)
    builder.H += sparse(SpinHalfOperator(builder.basis, bonds; J, sector))
    return builder
end

"""Add fields sum_i h⋅S_i in x,y,z order."""
function add_spin_field!(builder::SpinHalfBuilder; h, sites=1:builder.basis.N, sector::Symbol=:auto)
    builder.H += sparse(SpinHalfOperator(builder.basis; fields=[(i,h) for i in sites], sector))
    return builder
end

"""Build a sparse small-system check of arbitrary multi-site spin terms."""
function add_spin_term!(builder::SpinHalfBuilder, terms; sector::Symbol=:auto)
    builder.H += sparse(SpinHalfTermOperator(builder.basis, terms; sector))
    return builder
end

"""Sparse Heisenberg Hamiltonian in a full or fixed-Sz spin basis."""
heisenberg_hamiltonian(basis::SpinHalfBasis, bonds; J=1.0) =
    sparse(SpinHalfOperator(basis, bonds; J=J * Matrix{Float64}(I,3,3)))
