"""Independent tensor/Jordan-Wigner reference for at most four Hubbard sites."""
function tensor_hubbard(N, bonds; t=1.0, U=8.0, mu=4.0)
    N <= 4 || throw(ArgumentError("test reference is limited to four sites"))
    op = fermion_operators(use_sparse=true)
    builder = HamiltonianBuilder(OpenSqua(N,1),op;use_sparse=true)
    add_bond!(builder,op.a_up;coeff=-t,bonds,F=op.F,jw=true)
    add_bond!(builder,op.a_down;coeff=-t,bonds,F=op.F,jw=true)
    add_onsite!(builder,op.n_up*op.n_down;coeff=U)
    add_onsite!(builder,op.n_up+op.n_down;coeff=-mu)
    number = zeros(4^N)
    for i in 1:N
        number .+= diag(get_multi_site_op(Dict(i=>op.n_up+op.n_down),N;use_sparse=true))
    end
    return Matrix(matrix(builder)),number
end

function to_bit_basis(A, basis::SpinHalfBasis)
    indices = [1 + sum((1-Int((state >> (site-1)) & 1))*2^(basis.N-site)
                       for site in 1:basis.N) for state in basis.states]
    return Matrix(A[indices,indices])
end

@testset "Local spin and fermion algebra" begin
    for d in 2:4
        s = spin_operators(d)
        @test s.Sx*s.Sy-s.Sy*s.Sx ≈ im*s.Sz
        @test s.S² ≈ s.spin*(s.spin+1)*I
    end
    f = fermion_operators(use_sparse=true)
    operators = [jw_operator(c,site,2,f.F;use_sparse=true)
                 for c in (f.a_up,f.a_down) for site in 1:2]
    identity = Matrix{Float64}(I,16,16)
    for i in eachindex(operators), j in eachindex(operators)
        a,b = operators[i],operators[j]
        @test norm(a*b+b*a) < 1e-12
        @test norm(a*b'+b'*a-(i == j ? identity : zeros(16,16))) < 1e-12
    end
end
