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
        @test a ≈ b rtol = (Sleipnir.doublePrec ? 1e-12 : 1e-5)
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
        @test a ≈ b rtol=(Sleipnir.doublePrec ? 1e-12 : 1e-5)
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
    η₀ = 1.

    # Test clamp_borders_dx
    for i in range(1, 5)
        H = abs.(randn(size...)) # Implementation of the adjoint doesn't hold if H is negative
        dS = randn(size[1] - 1, size[2] - 2)
        v = randn(size[1] - 1, size[2] - 2)
        c = ODINN.clamp_borders_dx(dS, H, η₀, Δ)
        ∂dS = zero(dS)
        ∂H = zero(H)
        ODINN.clamp_borders_dx_adjoint!(∂dS, ∂H, v, η₀, Δ, H, dS)
        a=sum(c.*v)/fac
        b=sum(H.*∂H)/fac + sum(dS.*∂dS) / fac
        @test a≈b rtol=(Sleipnir.doublePrec ? 1e-12 : 1e-5)
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
        a=sum(c.*v)/fac
        b=sum(H.*∂H)/fac + sum(dS.*∂dS)/fac
        @test a≈b rtol=(Sleipnir.doublePrec ? 1e-12 : 1e-5)
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
        v = randn(size[1]-1,size[2]-1)
        Au = zeros(size[1]-1,size[2]-1)
        Huginn.avg!(Au, u)
        Aadjv = ODINN.avg_adjoint(v)
        a=sum(Au.*v)/fac
        b=sum(u.*Aadjv)/fac
        @test a≈b rtol=(Sleipnir.doublePrec ? 1e-12 : 1e-5)
    end

    # Test avg_x
    for i in range(1, 5)
        u = randn(size...)
        v = randn(size[1]-1,size[2])
        Au = zeros(size[1]-1,size[2])
        Huginn.avg_x!(Au, u)
        Aadjv = ODINN.avg_x_adjoint(v)
        a=sum(Au.*v)/fac
        b=sum(u.*Aadjv)/fac
        @test a≈b rtol=(Sleipnir.doublePrec ? 1e-12 : 1e-5)
    end

    # Test avg_y
    for i in range(1, 5)
        u = randn(size...)
        v = randn(size[1],size[2]-1)
        Au = zeros(size[1],size[2]-1)
        Huginn.avg_y!(Au, u)
        Aadjv = ODINN.avg_y_adjoint(v)
        a=sum(Au.*v)/fac
        b=sum(u.*Aadjv)/fac
        @test a≈b rtol=(Sleipnir.doublePrec ? 1e-12 : 1e-5)
    end
end

# """
#     test_adjoint_SIAD2D()

# Check the consistency between the forward and the adjoint of SIA2D.
# It uses finite differences to compute the gradient and it compares the obtained
# values with the ones computed using the discrete adjoint of SIA2D.
# """
# function test_adjoint_SIAD2D()
#     Random.seed!(1234)
#     function _loss(H, simulation, t, vecBackwardSIA2D)
#         dH = Huginn.SIA2D(H, simulation, t)
#         return sum(dH.*vecBackwardSIA2D)
#     end

#     nx = 10
#     ny = 12
#     tspan = (2010.0, 2015.0)
#     A = 4e-17
#     n = 3.0
#     Δx = 1.
#     Δy = 1.3
#     params = Huginn.Parameters(
#         simulation = SimulationParameters(
#             use_MB=false,
#             velocities=false,
#             tspan=tspan,
#             working_dir = Huginn.root_dir,
#             test_mode = true,
#         ),
#         solver = SolverParameters(reltol=1e-12)
#     )

#     model = Huginn.Model(iceflow = SIA2Dmodel(params), mass_balance = nothing)

#     H₀ = abs.(randn(nx, ny))
#     B = abs.(randn(nx, ny))
#     S = H₀ + B
#     glacier = Glacier2D(rgi_id = "toy", H₀ = H₀, S = S, B = B, A = A, n=n, Δx=Δx, Δy=Δy, nx=nx, ny=ny, C = 0.0)
#     glaciers = [glacier]

#     simulation = Prediction(model, glaciers, params)

#     glacier_idx = 1
#     initialize_iceflow_model(model.iceflow, glacier_idx, glaciers[glacier_idx], params)

#     H = H₀
#     t = tspan[1]
#     simulation.model.iceflow.glacier_idx = glacier_idx

#     vecBackwardSIA2D = randn(size(H,1), size(H,2))

#     dH = Huginn.SIA2D(H, simulation, t)

#     ∂H, ∂A = VJP_λ_∂SIA_discrete(vecBackwardSIA2D, H, simulation, t)

#     # Check gradient wrt H
#     function f_H(H, args)
#         simulation, t, vecBackwardSIA2D = args
#         return _loss(H, simulation, t, vecBackwardSIA2D)
#     end
#     ratio = []
#     angle = []
#     relerr = []
#     eps = []
#     for k in range(2,9)
#         ϵ = 10.0^(-k)
#         push!(eps, ϵ)
#         ∂H_num = compute_numerical_gradient(H, (simulation, t, vecBackwardSIA2D), f_H, ϵ; varStr="of H")
#         ratio_k, angle_k, relerr_k = stats_err_arrays(∂H, ∂H_num)
#         push!(ratio, ratio_k)
#         push!(angle, angle_k)
#         push!(relerr, relerr_k)
#     end
#     min_ratio = minimum(abs.(ratio))
#     min_angle = minimum(abs.(angle))
#     min_relerr = minimum(abs.(relerr))
#     thres_ratio = 1e-6
#     thres_angle = 1e-12
#     thres_relerr = 1e-7
#     if printDebug | !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
#         println("Gradient wrt H")
#         println("eps    = ",printVecScientific(eps))
#         printVecScientific("ratio  = ",ratio,thres_ratio)
#         printVecScientific("angle  = ",angle,thres_angle)
#         printVecScientific("relerr = ",relerr,thres_relerr)
#     end
#     @test (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr)

#     # Check gradient wrt A
#     function f_A(A, args)
#         H, simulation, t, vecBackwardSIA2D = args
#         simulation.model.iceflow.A[] = A[1]
#         return _loss(H, simulation, t, vecBackwardSIA2D)
#     end
#     ratio = []
#     angle = []
#     relerr = []
#     eps = []
#     for k in range(17,21)
#         ϵ = 10.0^(-k)
#         push!(eps, ϵ)
#         Avec = [simulation.model.iceflow.A[]]
#         ∂A_num = compute_numerical_gradient(Avec, (H, simulation, t, vecBackwardSIA2D), f_A, ϵ; varStr="of A")
#         ratio_k, angle_k, relerr_k = stats_err_arrays([∂A], ∂A_num)
#         push!(ratio, ratio_k)
#         push!(angle, angle_k)
#         push!(relerr, relerr_k)
#     end
#     min_ratio = minimum(abs.(ratio))
#     min_angle = minimum(abs.(angle))
#     min_relerr = minimum(abs.(relerr))
#     thres_ratio = 1e-14
#     thres_angle = 1e-14
#     thres_relerr = 1e-14
#     if printDebug | !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
#         println("Gradient wrt θ")
#         println("eps    = ",printVecScientific(eps))
#         printVecScientific("ratio  = ",ratio,thres_ratio)
#         printVecScientific("angle  = ",angle,thres_angle)
#         printVecScientific("relerr = ",relerr,thres_relerr)
#     end
#     @test (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr)
# end
