
function test_adjoint_SIAD2D_continuous()
    Random.seed!(1234)
    function _loss(H, θ, simulation, t, vecBackwardSIA2D)
        ODINN.apply_UDE_parametrization!(θ, simulation, nothing, glacier_idx)
        dH = Huginn.SIA2D(H, simulation, t; batch_id=glacier_idx)
        return sum(dH.*vecBackwardSIA2D)
    end

    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()

    working_dir = joinpath(ODINN.root_dir, "test/data")

    δt = 1/12
    tspan = (2010.0, 2015.0)

    params = Parameters(
        simulation = SimulationParameters(
            working_dir=working_dir,
            use_MB=false,
            velocities=true,
            tspan=tspan,
            step=δt,
            multiprocessing=false,
            workers=1,
            light=false, # for now we do the simulation like this (a better name would be dense)
            test_mode=true,
            rgi_paths=rgi_paths),
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-18),
        UDE = UDEparameters(
            sensealg=SciMLSensitivity.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=ContinuousAdjoint(),
            optimization_method="AD+AD",
            target = "A"),
        solver = Huginn.SolverParameters(
            step=δt,
            save_everystep=true,
            progress=true)
    )


    model = ODINN.Model(iceflow = SIA2Dmodel(params), mass_balance = nothing, machine_learning = NeuralNetwork(params))

    glaciers = initialize_glaciers(rgi_ids, params)

    glacier_idx = 1
    batch_idx = 1


    H = glaciers[glacier_idx].H₀

    simulation = FunctionalInversion(model, glaciers, params)

    initialize_iceflow_model(model.iceflow[glacier_idx], glacier_idx, glaciers[glacier_idx], params)

    t = tspan[1]
    θ = simulation.model.machine_learning.θ
    simulation.model.iceflow[batch_idx].glacier_idx = glacier_idx

    vecBackwardSIA2D = randn(size(H,1), size(H,2))

    # Initialize A by making one prediction with the neural network
    ODINN.apply_UDE_parametrization!(θ, simulation, nothing, glacier_idx)
    dH = Huginn.SIA2D(H, simulation, t; batch_id=batch_idx)

    ∂H = VJP_λ_∂SIA∂H_continuous(vecBackwardSIA2D, H, simulation, t; batch_id=batch_idx)
    ∂θ = VJP_λ_∂SIA∂θ_continuous(θ, vecBackwardSIA2D, H, simulation, t; batch_id=batch_idx)

    # Check gradient wrt H
    function f_H(H, args)
        simulation, t, vecBackwardSIA2D = args
        θ = simulation.model.machine_learning.θ
        return _loss(H, θ, simulation, t, vecBackwardSIA2D)
    end
    ratio = []
    angle = []
    relerr = []
    eps = []
    for k in range(3,8)
        ϵ = 10.0^(-k)
        push!(eps, ϵ)
        ∂H_num = compute_numerical_gradient(H, (simulation, t, vecBackwardSIA2D), f_H, ϵ; varStr="of H")
        ratio_k, angle_k, relerr_k = stats_err_arrays(∂H, ∂H_num)
        push!(ratio, ratio_k)
        push!(angle, angle_k)
        push!(relerr, relerr_k)
    end
    min_ratio = minimum(abs.(ratio))
    min_angle = minimum(abs.(angle))
    min_relerr = minimum(abs.(relerr))
    thres_ratio = 1e-4
    thres_angle = 2e-4
    thres_relerr = 2e-2
    if !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
        println("Gradient wrt H")
        println("eps    = ",printVecScientific(eps))
        println("ratio  = ",printVecScientific(ratio))
        println("angle  = ",printVecScientific(angle))
        println("relerr = ",printVecScientific(relerr))
    end
    @test min_ratio<thres_ratio
    @test min_angle<thres_angle
    @test min_relerr<thres_relerr

    # Check gradient wrt θ
    function f_θ(θ, args)
        H, simulation, t, vecBackwardSIA2D = args
        return _loss(H, θ, simulation, t, vecBackwardSIA2D)
    end
    ratio = []
    angle = []
    relerr = []
    eps = []
    for k in range(5,7)
        ϵ = 10.0^(-k)
        push!(eps, ϵ)
        ∂θ_num = Huginn.compute_numerical_gradient(θ, (H, simulation, t, vecBackwardSIA2D), f_θ, ϵ; varStr="of θ")
        ratio_k, angle_k, relerr_k = Huginn.stats_err_arrays(∂θ, ∂θ_num)
        push!(ratio, ratio_k)
        push!(angle, angle_k)
        push!(relerr, relerr_k)
    end
    min_ratio = minimum(abs.(ratio))
    min_angle = minimum(abs.(angle))
    min_relerr = minimum(abs.(relerr))
    thres_ratio = 3e-2
    thres_angle = 1e-14
    thres_relerr = 3e-2
    if !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
        println("Gradient wrt θ")
        println("eps    = ",printVecScientific(eps))
        println("ratio  = ",printVecScientific(ratio))
        println("angle  = ",printVecScientific(angle))
        println("relerr = ",printVecScientific(relerr))
    end
    @test (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr)
end
