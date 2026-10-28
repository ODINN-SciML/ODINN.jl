"""
    test_adjoint_diff()

Check the consistency between the forward and the adjoint of the diff functions.
It uses the definition of the adjoint of an operator, that is
<u,Av>=<A^*u,v> for all u,v
"""
function test_adjoint_diff()
    size = (10, 11)
    fac = prod(size)
    Δ = 2.5

    # Test diff_x
    for i in range(1, 5)
        u = randn(size...)
        v = randn(size[1] - 1, size[2])
        Au = zeros(size[1] - 1, size[2])
        Huginn.diff_x!(Au, u, Δ)
        Aadjv = ODINN.diff_x_adjoint(v, Δ)
        a = sum(Au .* v) / fac
        b = sum(u .* Aadjv)/fac
        @test a ≈ b rtol = (Sleipnir.doublePrec ? 1e-11 : 1e-5)
    end

    # Test diff_y
    for i in range(1, 5)
        u = randn(size...)
        v = randn(size[1], size[2] - 1)
        Au = zeros(size[1], size[2] - 1)
        Huginn.diff_y!(Au, u, Δ)
        Aadjv = ODINN.diff_y_adjoint(v, Δ)
        a = sum(Au .* v) / fac
        b = sum(u .* Aadjv) / fac
        @test a ≈ b rtol=(Sleipnir.doublePrec ? 1e-11 : 1e-5)
    end
end

"""
    test_adjoint_clamp_borders()

Check the consistency between the forward and the adjoint of the clamp functions.
It uses the definition of the adjoint of an operator, that is
<u,Av>=<A^*u,v> for all u,v
"""
function test_adjoint_clamp_borders()
    size = (10, 11)
    fac = prod(size)
    Δ = 2.5
    η₀ = 1.0

    # Test clamp_borders_dx
    for i in range(1, 5)
        H = abs.(randn(size...)) # Implementation of the adjoint doesn't hold if H is negative
        dS = randn(size[1] - 1, size[2] - 2)
        v = randn(size[1] - 1, size[2] - 2)
        c = ODINN.clamp_borders_dx(dS, H, η₀, Δ)
        ∂dS = zero(dS)
        ∂H = zero(H)
        ODINN.clamp_borders_dx_adjoint!(∂dS, ∂H, v, η₀, Δ, H, dS)
        a=sum(c .* v)/fac
        b=sum(H .* ∂H)/fac + sum(dS .* ∂dS) / fac
        @test a≈b rtol=(Sleipnir.doublePrec ? 1e-11 : 1e-5)
    end

    # Test clamp_borders_dy
    for i in range(1, 5)
        H = abs.(randn(size...)) # Implementation of the adjoint doesn't hold if H is negative
        dS = randn(size[1]-2, size[2]-1)
        v = randn(size[1]-2, size[2]-1)
        c = ODINN.clamp_borders_dy(dS, H, η₀, Δ)
        ∂dS = zero(dS)
        ∂H = zero(H)
        ODINN.clamp_borders_dy_adjoint!(∂dS, ∂H, v, η₀, Δ, H, dS)
        a=sum(c .* v)/fac
        b=sum(H .* ∂H)/fac + sum(dS .* ∂dS)/fac
        @test a≈b rtol=(Sleipnir.doublePrec ? 1e-11 : 1e-5)
    end
end

"""
    test_laplacian()

Check the 5-point Laplacian `ODINN.∇²` and its adjoint `ODINN.VJP_λ_∂∇²a_∂a`: it is exact for a
quadratic field, a checkerboard is not in its null space (the former staggered version gave
exactly zero on it), and <λ, ∇²a> = <VJP(λ), a> for all a, λ (the operator is linear).
"""
function test_laplacian()
    Δx, Δy = 2.0, 3.0
    dims = (8, 7)
    x = [Δx * (i - 1) for i in 1:dims[1], j in 1:dims[2]]
    y = [Δy * (j - 1) for i in 1:dims[1], j in 1:dims[2]]
    L = ODINN.∇²(x .^ 2 .+ 2 .* y .^ 2, Δx, Δy)
    @test all(L[2:(end - 1), 2:(end - 1)] .≈ 6.0)
    @test all(L[[1, end], :] .== 0) && all(L[:, [1, end]] .== 0)

    cb = [(-1.0)^(i + j) for i in 1:dims[1], j in 1:dims[2]]
    Lcb = ODINN.∇²(cb, Δx, Δy)
    @test Lcb[2:(end - 1), 2:(end - 1)] ≈
          (-4 / Δx^2 - 4 / Δy^2) .* cb[2:(end - 1), 2:(end - 1)]

    for i in range(1, 5)
        a, λ = randn(dims...), randn(dims...)
        lhs = sum(λ .* ODINN.∇²(a, Δx, Δy))
        rhs = sum(ODINN.VJP_λ_∂∇²a_∂a(λ, a, Δx, Δy) .* a)
        @test lhs≈rhs rtol=(Sleipnir.doublePrec ? 1e-11 : 1e-5)
    end
end

"""
    test_adjoint_avg()

Check the consistency between the forward and the adjoint of the average functions.
It uses the definition of the adjoint of an operator, that is
<u,Av>=<A^*u,v> for all u,v
"""
function test_adjoint_avg()
    size = (10, 11)
    fac = prod(size)

    # Test avg
    for i in range(1, 5)
        u = randn(size...)
        v = randn(size[1]-1, size[2]-1)
        Au = zeros(size[1]-1, size[2]-1)
        Huginn.avg!(Au, u)
        Aadjv = ODINN.avg_adjoint(v)
        a=sum(Au .* v)/fac
        b=sum(u .* Aadjv)/fac
        @test a≈b rtol=(Sleipnir.doublePrec ? 1e-11 : 1e-5)
    end

    # Test avg_x
    for i in range(1, 5)
        u = randn(size...)
        v = randn(size[1]-1, size[2])
        Au = zeros(size[1]-1, size[2])
        Huginn.avg_x!(Au, u)
        Aadjv = ODINN.avg_x_adjoint(v)
        a=sum(Au .* v)/fac
        b=sum(u .* Aadjv)/fac
        @test a≈b rtol=(Sleipnir.doublePrec ? 1e-11 : 1e-5)
    end

    # Test avg_y
    for i in range(1, 5)
        u = randn(size...)
        v = randn(size[1], size[2]-1)
        Au = zeros(size[1], size[2]-1)
        Huginn.avg_y!(Au, u)
        Aadjv = ODINN.avg_y_adjoint(v)
        a=sum(Au .* v)/fac
        b=sum(u .* Aadjv)/fac
        @test a≈b rtol=(Sleipnir.doublePrec ? 1e-11 : 1e-5)
    end
end
