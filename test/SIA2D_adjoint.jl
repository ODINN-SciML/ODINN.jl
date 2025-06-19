
function test_adjoint_SIA2D(
    adjointFlavor::ADJ;
    thres = [2e-4, 2e-4, 2e-2],
    target = :A
) where {ADJ<:AbstractAdjointMethod}

    Random.seed!(1234)

    thres_ratio = thres[1]
    thres_angle = thres[2]
    thres_relerr = thres[3]

    function _loss(H, θ, simulation, t, vecBackwardSIA2D)
        apply_parametrization!(
            simulation.model.machine_learning.target;
            H = H, ∇S = nothing, θ = θ,
            iceflow_model = simulation.model.iceflow[batch_idx],
            ml_model = simulation.model.machine_learning,
            glacier = glaciers[batch_idx],
            params = simulation.parameters
        )
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
            test_mode=true,
            rgi_paths=rgi_paths),
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-18),
        UDE = UDEparameters(
            sensealg=SciMLSensitivity.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=adjointFlavor,
            optimization_method="AD+AD",
            target = target),
        solver = Huginn.SolverParameters(
            step=δt,
            save_everystep=true,
            progress=true)
    )


    model = ODINN.Model(
        iceflow = SIA2Dmodel(params),
        mass_balance = nothing,
        machine_learning = NeuralNetwork(params)
    )

    glaciers = initialize_glaciers(rgi_ids, params)

    glacier_idx = 1
    batch_idx = 1

    H = glaciers[glacier_idx].H₀

    simulation = FunctionalInversion(model, glaciers, params)

    initialize_iceflow_model(model.iceflow[glacier_idx], glacier_idx, glaciers[glacier_idx], params)

    t = tspan[1]
    θ = simulation.model.machine_learning.θ
    iceflow_model = simulation.model.iceflow[batch_idx]
    iceflow_model.glacier_idx = glacier_idx

    vecBackwardSIA2D = randn(size(H, 1), size(H, 2))

    # Initialize A by making one prediction with the neural network
    apply_parametrization!(
        simulation.model.machine_learning.target;
        H = H, ∇S = nothing, θ = θ,
        iceflow_model = iceflow_model,
        ml_model = simulation.model.machine_learning,
        glacier = glaciers[batch_idx],
        params = simulation.parameters
    )
    dH = Huginn.SIA2D(H, simulation, t; batch_id = batch_idx)

    ∂H, = ODINN.VJP_λ_∂SIA∂H(
        adjointFlavor.VJP_method,
        vecBackwardSIA2D,
        H,
        θ,
        simulation,
        t,
        batch_idx
        )
    ∂θ = ODINN.VJP_λ_∂SIA∂θ(
        adjointFlavor.VJP_method,
        vecBackwardSIA2D,
        H,
        θ,
        nothing,
        nothing,
        simulation,
        t,
        batch_idx
        )

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
    if printDebug | !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
        println("Gradient wrt H")
        println("eps    = ",printVecScientific(eps))
        printVecScientific("ratio  = ",ratio,thres_ratio)
        printVecScientific("angle  = ",angle,thres_angle)
        printVecScientific("relerr = ",relerr,thres_relerr)
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
        ∂θ_num = compute_numerical_gradient(θ, (H, simulation, t, vecBackwardSIA2D), f_θ, ϵ; varStr="of θ")
        ratio_k, angle_k, relerr_k = stats_err_arrays(∂θ, ∂θ_num)
        push!(ratio, ratio_k)
        push!(angle, angle_k)
        push!(relerr, relerr_k)
    end
    min_ratio = minimum(abs.(ratio))
    min_angle = minimum(abs.(angle))
    min_relerr = minimum(abs.(relerr))
    if printDebug | !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
        println("Gradient wrt θ")
        println("eps    = ",printVecScientific(eps))
        printVecScientific("ratio  = ",ratio,thres_ratio)
        printVecScientific("angle  = ",angle,thres_angle)
        printVecScientific("relerr = ",relerr,thres_relerr)
    end
    @test min_ratio<thres_ratio
    @test min_angle<thres_angle
    @test min_relerr<thres_relerr
end


function test_adjoint_surface_V(
    adjointFlavor::ADJ;
    thres = [2e-4, 2e-4, 2e-2],
    target = :A
) where {ADJ<:AbstractAdjointMethod}

    Random.seed!(1234)

    thres_ratio = thres[1]
    thres_angle = thres[2]
    thres_relerr = thres[3]

    function _loss(H, θ, simulation, t, vecBackwardSIA2D)
        apply_parametrization!(
            simulation.model.machine_learning.target;
            H = H, ∇S = nothing, θ = θ,
            iceflow_model = simulation.model.iceflow[batch_idx],
            ml_model = simulation.model.machine_learning,
            glacier = glaciers[batch_idx],
            params = simulation.parameters
        )
        Vx, Vy = Huginn.surface_V(H, simulation; batch_id=glacier_idx)
        return sum(Vx.*vecBackwardSIA2D[1]+Vy.*vecBackwardSIA2D[2])
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
            test_mode=true,
            rgi_paths=rgi_paths),
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-18),
        UDE = UDEparameters(
            sensealg=SciMLSensitivity.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=adjointFlavor,
            optimization_method="AD+AD",
            target = target),
        solver = Huginn.SolverParameters(
            step=δt,
            save_everystep=true,
            progress=true)
    )


    model = ODINN.Model(
        iceflow = SIA2Dmodel(params),
        mass_balance = nothing,
        machine_learning = NeuralNetwork(params)
    )

    glaciers = initialize_glaciers(rgi_ids, params)

    glacier_idx = 1
    batch_idx = 1

    H = glaciers[glacier_idx].H₀

    simulation = FunctionalInversion(model, glaciers, params)

    initialize_iceflow_model(model.iceflow[glacier_idx], glacier_idx, glaciers[glacier_idx], params)

    t = tspan[1]
    θ = simulation.model.machine_learning.θ
    iceflow_model = simulation.model.iceflow[batch_idx]
    iceflow_model.glacier_idx = glacier_idx

    vecBackwardSIA2D = [randn(size(H, 1)-1, size(H, 2)-1), randn(size(H, 1)-1, size(H, 2)-1)]

    # Initialize A by making one prediction with the neural network
    apply_parametrization!(
        simulation.model.machine_learning.target;
        H = H, ∇S = nothing, θ = θ,
        iceflow_model = iceflow_model,
        ml_model = simulation.model.machine_learning,
        glacier = glaciers[batch_idx],
        params = simulation.parameters
    )
    Vx, Vy = Huginn.surface_V(H, simulation; batch_id = batch_idx)

    ∂H, = ODINN.VJP_λ_∂surface_V∂H(
        adjointFlavor.VJP_method,
        vecBackwardSIA2D[1],
        vecBackwardSIA2D[2],
        H,
        θ,
        simulation,
        t,
        batch_idx
    )
    ∂θ, = ODINN.VJP_λ_∂surface_V∂θ(
        adjointFlavor.VJP_method,
        vecBackwardSIA2D[1],
        vecBackwardSIA2D[2],
        H,
        θ,
        nothing,
        nothing,
        simulation,
        t,
        batch_idx
    )

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
    if printDebug | !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
        println("Gradient wrt H")
        println("eps    = ",printVecScientific(eps))
        printVecScientific("ratio  = ",ratio,thres_ratio)
        printVecScientific("angle  = ",angle,thres_angle)
        printVecScientific("relerr = ",relerr,thres_relerr)
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
        ∂θ_num = compute_numerical_gradient(θ, (H, simulation, t, vecBackwardSIA2D), f_θ, ϵ; varStr="of θ")
        ratio_k, angle_k, relerr_k = stats_err_arrays(∂θ, ∂θ_num)
        push!(ratio, ratio_k)
        push!(angle, angle_k)
        push!(relerr, relerr_k)
    end
    min_ratio = minimum(abs.(ratio))
    min_angle = minimum(abs.(angle))
    min_relerr = minimum(abs.(relerr))
    if printDebug | !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
        println("Gradient wrt θ")
        println("eps    = ",printVecScientific(eps))
        printVecScientific("ratio  = ",ratio,thres_ratio)
        printVecScientific("angle  = ",angle,thres_angle)
        printVecScientific("relerr = ",relerr,thres_relerr)
    end
    @test min_ratio<thres_ratio
    @test min_angle<thres_angle
    @test min_relerr<thres_relerr
end
