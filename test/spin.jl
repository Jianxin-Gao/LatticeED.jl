@testset "Small spin operators against tensor products" begin
    basis = SpinHalfBasis(3)
    local_ops = spin_operators()
    ops = (local_ops.Sx,local_ops.Sy,local_ops.Sz)
    x = randn(Xoshiro(9),ComplexF64,8)
    for axes in ((1,1,1),(2,2,2),(3,3,3),(1,2,3),(1,2,2))
        tensor = get_multi_site_op(Dict(i=>ops[axes[i]] for i in 1:3),3)
        reference = to_bit_basis(tensor,basis)
        op = SpinHalfTermOperator(basis,[((1,2,3),axes,1)])
        @test Matrix(op) ≈ reference atol=1e-12
        @test op*x ≈ reference*x atol=1e-12
        @test Matrix(sparse(op)) ≈ reference atol=1e-12
    end

    real_op = SpinHalfOperator(basis,[(1,2),(2,3)];J=Matrix{Float64}(I,3,3))
    complex_op = SpinHalfOperator(basis;fields=[(1,[0,1,0])])
    @test eltype(real_op) == Float64
    @test eltype(complex_op) == ComplexF64
    @test real_op*x ≈ Matrix(real_op)*x
    @test complex_op*x ≈ to_bit_basis(get_multi_site_op(Dict(1=>local_ops.Sy),3),basis)*x

    tensor_builder = HamiltonianBuilder(OpenSqua(3,1),local_ops;use_sparse=true)
    for component in ops
        add_bond!(tensor_builder,component;coeff=0.7,bonds=[(1,2),(2,3)])
    end
    add_onsite!(tensor_builder,local_ops.Sz;coeff=-0.3)
    for nup in 0:3
        sector = SpinHalfBasis(3,nup)
        expected = to_bit_basis(matrix(tensor_builder),sector)
        op = SpinHalfOperator(sector,[(1,2),(2,3)];J=0.7*Matrix{Float64}(I,3,3),
                              fields=[(i,[0,0,-0.3]) for i in 1:3])
        builder = SpinHalfBuilder(sector)
        add_spin_bond!(builder,[(1,2),(2,3)];J=0.7*Matrix{Float64}(I,3,3))
        add_spin_field!(builder;h=[0,0,-0.3])
        @test Matrix(op) ≈ expected atol=1e-12
        @test Matrix(matrix(builder)) ≈ expected atol=1e-12
    end
end

@testset "Two-site Heisenberg energy and magnetization" begin
    tensor = heisenberg_hamiltonian(OpenSqua(2,1),1,spin_operators())
    @test eigvals(Hermitian(Matrix(tensor))) ≈ [-0.75,0.25,0.25,0.25] atol=1e-12
    beta = [0.0,0.25,1.0,2.0]
    blocks = heisenberg_blocks(2,[(1,2)];hz=0.3)
    @test sum(b.dim for b in blocks) == 4
    records = ftlm_samples(blocks;sampling=:basis,M=2,reorthogonalize=true)
    energy = thermal_energy(records,beta;nsites=2)
    exact = exact_energy(blocks,beta;nsites=2)
    @test energy.energy_per_site ≈ exact.energy_per_site atol=1e-12
    @test energy.logZ ≈ exact.logZ atol=1e-12
    @test all(iszero,energy.energy_errors)
    scalars = Dict(b.key=>total_Sz(b) for b in blocks)
    magnetization = thermal_block_observable(records,energy,scalars;normalization=2)
    direct = thermal_block_observable(records,beta,scalars;normalization=2)
    @test magnetization.values ≈ direct.values atol=1e-12
    # Independent singlet/triplet levels in the applied field.
    es,ms = [-0.75,0.25-0.3,0.25,0.25+0.3],[0.0,1.0,0.0,-1.0]
    for (i,b) in enumerate(beta)
        w = exp.(-b.*(es.-minimum(es)))
        @test energy.energy_per_site[i] ≈ dot(w,es)/sum(w)/2 atol=1e-12
        @test magnetization.values[i] ≈ dot(w,ms)/sum(w)/2 atol=1e-12
    end
end

@testset "Small exact thermal averages" begin
    H = [-2.0 0.4;0.4 1.0]
    F = eigen(Hermitian(H))
    beta = [0.0,0.5,1.0,2.0]
    average = thermal_expectation(F.values,F.vectors,H,beta)
    @test real.(average) ≈ exact_energy([H],beta).values atol=1e-12
    state = ComplexF64[1,im]/sqrt(2)
    @test thermal_expectation(F.values,F.vectors,H,0.0;mode=:state,state)[1] ≈ dot(state,H*state)
end
