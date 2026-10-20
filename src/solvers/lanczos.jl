"""
    lanczos(H, v0; m=100, tol=1e-12, reorthogonalize=true,
            store_vectors=reorthogonalize)

Hermitian Lanczos with real/complex type preservation. Full reorthogonalization
uses two modified Gram–Schmidt passes. Low-memory mode keeps three work vectors
and returns `V=nothing`; it must be checked for convergence in m.
The returned V is a view of the used columns, not a second basis allocation.
"""
function lanczos(H, v0::AbstractVector; m::Int=100, tol::Real=1e-12,
                 reorthogonalize::Bool=true, store_vectors::Bool=reorthogonalize)
    n = length(v0)
    n > 0 && size(H) == (n, n) || throw(DimensionMismatch("Lanczos requires a nonempty square operator"))
    m > 0 || throw(ArgumentError("m must be positive"))
    isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
    reorthogonalize && !store_vectors && throw(ArgumentError("full reorthogonalization requires storing vectors"))
    T = promote_type(Float64, eltype(H), eltype(v0))
    v = Vector{T}(v0)
    vnorm = norm(v)
    isfinite(vnorm) && vnorm > 0 || throw(ArgumentError("v0 must be finite and nonzero"))
    v ./= vnorm
    mmax = min(m, n)
    V = store_vectors ? Matrix{T}(undef, n, mmax) : nothing
    α, β = Float64[], Float64[]
    previous, work = zeros(T, n), similar(v)
    for k in 1:mmax
        store_vectors && copyto!(view(V, :, k), v)
        mul!(work, H, v)
        scale = norm(work)
        isfinite(scale) || throw(ArgumentError("nonfinite operator action"))
        k > 1 && axpy!(-β[end], previous, work)
        a = dot(v, work)
        abs(imag(a)) <= 1e-10 * max(1, scale) || throw(ArgumentError("Lanczos requires a Hermitian operator"))
        push!(α, real(a))
        axpy!(-real(a), v, work)
        if reorthogonalize
            for _ in 1:2, q in 1:k
                vq = view(V, :, q)
                axpy!(-dot(vq, work), vq, work)
            end
        end
        b = norm(work)
        if b <= tol * max(1, scale) || k == mmax
            return (α=α, β=β, V=store_vectors ? view(V, :, 1:k) : nothing)
        end
        push!(β, b)
        # Rotate existing buffers instead of allocating/copying a new vector.
        previous, v, work = v, work, previous
        v ./= b
    end
    error("unreachable")
end
