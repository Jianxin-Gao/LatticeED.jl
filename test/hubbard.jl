@testset "Three-site Hubbard sectors and fermion signs" begin
    bonds = [(1,2,0.7),(2,3,-0.2),(1,3,1.3)]
    reference,_ = tensor_hubbard(3,bonds;t=0.9,U=3.1,mu=0.8)
    spectrum = Float64[]
    for nu in 0:3, nd in 0:3
        basis = HubbardBasis(3,nu,nd)
        @test length(basis) == binomial(3,nu)*binomial(3,nd)
        @test all(state_index(basis,basis[i]) == i for i in 1:length(basis))
        H = HubbardOperator(basis,bonds;t=0.9,U=3.1,mu=0.8)
        indices,phases = tensor_basis_map(basis)
        expected = reference[indices,indices].*(phases*phases')
        @test Matrix(H) ≈ expected atol=1e-12
        @test ishermitian(H)
        x = randn(Xoshiro(11),length(basis))
        @test H*x ≈ expected*x atol=1e-12
        z = complex.(x,reverse(x))
        @test H*z ≈ expected*z atol=1e-12
        append!(spectrum,eigvals(Hermitian(Matrix(H))))
    end
    @test sort(spectrum) ≈ eigvals(Hermitian(reference)) atol=1e-12
end

function independent_exact_2x2(beta; t=1.0,U=8.0,mu=4.0)
    H,number = tensor_hubbard(4,square_bonds(2,2);t,U,mu)
    F = eigen(Hermitian(H))
    particles = vec(sum(abs2.(F.vectors).*number;dims=1))
    physical,grand,logZ = Float64[],Float64[],Float64[]
    for b in beta
        w = exp.(-b.*(F.values.-minimum(F.values)))
        z = sum(w)
        k = dot(w,F.values)/z
        n = dot(w,particles)/z
        push!(physical,(k+mu*n)/4)
        push!(grand,k/4)
        push!(logZ,-b*minimum(F.values)+log(z))
    end
    return (energy_per_site=physical,grand_energy_per_site=grand,logZ=logZ)
end

@testset "2x2 open Hubbard thermal energy" begin
    beta = [0.0,0.25,0.5,1.0,2.0]
    reference = independent_exact_2x2(beta)
    blocks = hubbard_blocks(4,square_bonds(2,2))
    @test sum(b.dim*b.multiplicity for b in blocks) == 4^4
    records = ftlm_samples(blocks;sampling=:basis,M=36,reorthogonalize=true)
    energy = thermal_energy(records,beta;nsites=4)
    exact = exact_energy(blocks,beta;nsites=4)
    for result in (energy,exact)
        @test result.energy_per_site ≈ reference.energy_per_site atol=1e-11
        @test result.grand_energy_per_site ≈ reference.grand_energy_per_site atol=1e-11
        @test result.logZ ≈ reference.logZ atol=1e-10
        @test result.density ≈ ones(length(beta)) atol=1e-12
        @test all(iszero,result.energy_errors)
    end
    @test energy.energy_per_site[1] ≈ 2.0 atol=1e-12
    @test energy.grand_energy_per_site[1] ≈ -2.0 atol=1e-12
end

@testset "Two-site canonical Hubbard trace" begin
    blocks = hubbard_blocks(2,[(1,2)];ensemble=:canonical,Ne=1,mu=0.7)
    @test sum(b.dim*b.multiplicity for b in blocks) == binomial(4,1)
    beta = [0.0,0.5,1.0,2.0]
    result = exact_energy(blocks,beta;nsites=2)
    H,number = tensor_hubbard(2,[(1,2)];mu=0.7)
    indices = findall(==(1),number)
    spectrum = eigvals(Hermitian(H[indices,indices]))
    for (i,b) in enumerate(beta)
        w = exp.(-b.*(spectrum.-minimum(spectrum)))
        @test result.grand_energy_per_site[i] ≈ dot(w,spectrum)/sum(w)/2 atol=1e-12
    end
    @test result.density ≈ fill(0.5,length(beta)) atol=1e-12
    @test result.energy_per_site-result.grand_energy_per_site ≈ fill(0.35,length(beta)) atol=1e-12
end

@testset "Two-site analytic Hubbard references" begin
    beta = [0.0,0.5,1.0,2.0]
    atomic = exact_energy(hubbard_blocks(2,[(1,2)];t=0),beta;nsites=2)
    @test atomic.energy_per_site ≈ 4 ./ (1 .+ exp.(4 .* beta)) atol=1e-12
    @test atomic.logZ ≈ 2 .* (log(2) .+ 4 .* beta .+ log1p.(exp.(-4 .* beta))) atol=1e-12

    mu = 0.7
    free = exact_energy(hubbard_blocks(2,[(1,2)];U=0,mu),beta;nsites=2)
    spectrum = [-1.0,1.0]
    for (i,b) in enumerate(beta)
        occupation = 1 ./ (1 .+ exp.(b.*(spectrum.-mu)))
        @test free.energy_per_site[i] ≈ dot(spectrum,occupation) atol=1e-12
        @test free.particles[i] ≈ 2*sum(occupation) atol=1e-12
        @test free.logZ[i] ≈ 2*sum(log1p.(exp.(-b.*(spectrum.-mu)))) atol=1e-12
    end
end
