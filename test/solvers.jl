@testset "Real and complex Lanczos" begin
    real_H = [1.0 0.3;0.3 -2.0]
    complex_H = ComplexF64[1 im;-im 3]
    for H in (real_H,complex_H)
        full = lanczos(H,[1.0,2.0];m=2,reorthogonalize=true)
        low = lanczos(H,[1.0,2.0];m=2,reorthogonalize=false)
        @test eigvals(SymTridiagonal(full.α,full.β)) ≈ eigvals(H)
        @test eigvals(SymTridiagonal(low.α,low.β)) ≈ eigvals(H)
        @test full.V'*full.V ≈ I
        @test H*full.V ≈ full.V*SymTridiagonal(full.α,full.β)
        @test low.V === nothing
    end
end

@testset "Small FTLM traces" begin
    H = [1.0 0.3;0.3 -2.0]
    beta = [0.0,0.5,1.0,2.0]
    basis = ftlm_energy(H,beta;sampling=:basis,M=2,reorthogonalize=true)
    exact = exact_energy([H],beta)
    @test basis.values ≈ exact.values atol=1e-12
    @test basis.logZ ≈ exact.logZ atol=1e-12
    @test all(iszero,basis.errors)

    # A short paired run checks the random path without a statistical sweep.
    low = ftlm_energy(H,beta;R=4,M=2,seed=77)
    full = ftlm_energy(H,beta;R=4,M=2,seed=77,reorthogonalize=true)
    @test low.values ≈ full.values atol=1e-12
    @test low.errors ≈ full.errors atol=1e-12
    @test all(isfinite,low.errors) && all(>=(0),low.errors)
    @test all(eigmin(H) <= e <= eigmax(H) for e in low.values)
    @test low.logZ[1] ≈ log(2) atol=1e-12
end
