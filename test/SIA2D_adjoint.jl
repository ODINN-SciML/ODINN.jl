
function test_adjoint_SIA2D(
    adjointFlavor::ADJ;
    thres = [2e-4, 2e-4, 2e-2],
    target = :A,
    C = 0.0,
    functional_inv = true,
    scalar = true,
) where {ADJ<:AbstractAdjointMethod}

    Random.seed!(1234)

    thres_ratio = thres[1]
    thres_angle = thres[2]
    thres_relerr = thres[3]

    function _loss(H, θ, simulation, t, vecBackwardSIA2D, glacier_idx)
        dH = zero(H)
        apply_all_callback_laws!(simulation.model.iceflow, simulation.cache.iceflow, simulation, glacier_idx, t, θ)
        Huginn.SIA2D!(dH, H, simulation, t, θ)
        return sum(dH.*vecBackwardSIA2D)
    end

    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()

    working_dir = joinpath(ODINN.root_dir, "test/data")

    δt = 1/12
    tspan = (2010.0, 2012.0)

    params = Parameters(
        simulation = SimulationParameters(
            working_dir=working_dir,
            use_MB=false,
            use_velocities=true,
            tspan=tspan,
            multiprocessing=false,
            test_mode=true,
            rgi_paths=rgi_paths,
            gridScalingFactor = (functional_inv || scalar) ? 1 : 4),
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-17),
        UDE = UDEparameters(
            sensealg=SciMLSensitivity.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=adjointFlavor,
            optimization_method="AD+AD",
            target = target),
        solver = Huginn.SolverParameters(
            step=δt,
            progress=true)
    )

    glaciers = initialize_glaciers(rgi_ids, params)

    trainable_model = if functional_inv
        NeuralNetwork(params)
    elseif scalar
        GlacierWideInv(params, glaciers, target)
    else
        GriddedInv(params, glaciers, target)
    end

    model = if target==:A
        law = if functional_inv
            LawA(trainable_model, params; precompute_VJPs=false, scalar=scalar)
        else
            LawA(params; scalar=scalar)
        end
        iceflow_model = SIA2Dmodel(params; A=law)
        Model(
            iceflow = iceflow_model,
            mass_balance = nothing,
            regressors = (; A=trainable_model),
        )
    elseif target==:D_hybrid
        @assert functional_inv
        iceflow_model = SIA2Dmodel(params; Y=LawY(trainable_model, params))
        Model(
            iceflow = iceflow_model,
            mass_balance = nothing,
            regressors = (; A=trainable_model),
        )
    elseif target==:D
        @assert functional_inv
        iceflow_model = SIA2Dmodel(params; U=LawU(trainable_model, params))
        Model(
            iceflow = iceflow_model,
            mass_balance = nothing,
            regressors = (; U=trainable_model),
        )
    else
        throw("Unsupported target $(target)")
    end

    glacier_idx = 1

    H = glaciers[glacier_idx].H₀
    glaciers[glacier_idx].C = C

    simulation = Inversion(model, glaciers, params)

    t = tspan[1]
    θ = simulation.model.trainable_components.θ
    cache = init_cache(model, simulation, glacier_idx, θ)
    simulation.cache = cache

    vecBackwardSIA2D = randn(size(H, 1), size(H, 2))

    # If the law being tested is a callback law, we need to apply it first
    apply_all_callback_laws!(model.iceflow, cache.iceflow, simulation, glacier_idx, t, θ)
    dH = zero(H)
    Huginn.SIA2D!(dH, H, simulation, t, θ)
    JET.@test_opt broken=true target_modules=(Sleipnir, Muninn, Huginn, ODINN) Huginn.SIA2D!(dH, H, simulation, t, θ)

    precompute_all_VJPs_laws!(model.iceflow, cache.iceflow, simulation, glacier_idx, t, θ)
    ∂H, = ODINN.VJP_λ_∂SIA∂H(
        adjointFlavor.VJP_method,
        vecBackwardSIA2D,
        H,
        θ,
        simulation,
        t,
    )
    ∂θ = ODINN.VJP_λ_∂SIA∂θ(
        adjointFlavor.VJP_method,
        vecBackwardSIA2D,
        H,
        θ,
        nothing,
        simulation,
        t,
    )

    # Check gradient wrt H
    function f_H(H, args)
        simulation, t, vecBackwardSIA2D, glacier_idx = args
        θ = simulation.model.trainable_components.θ
        return _loss(H, θ, simulation, t, vecBackwardSIA2D, glacier_idx)
    end
    ratio = []
    angle = []
    relerr = []
    eps = []
    for k in range(3,7,step=2)
        ϵ = 10.0^(-k)
        push!(eps, ϵ)
        ∂H_num = compute_numerical_gradient(H, (simulation, t, vecBackwardSIA2D, glacier_idx), f_H, ϵ; varStr="of H")
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
        H, simulation, t, vecBackwardSIA2D, glacier_idx = args
        return _loss(H, θ, simulation, t, vecBackwardSIA2D, glacier_idx)
    end
    ratio = []
    angle = []
    relerr = []
    eps = []
    for k in range(3,7)
        ϵ = 10.0^(-k)
        push!(eps, ϵ)
        ∂θ_num = compute_numerical_gradient(θ, (H, simulation, t, vecBackwardSIA2D, glacier_idx), f_θ, ϵ; varStr="of θ")
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

    function _loss(H, θ, simulation, t, vecBackwardSIA2D, glacier_idx)
        simulation.model.trainable_components.θ = θ
        apply_all_callback_laws!(simulation.model.iceflow, simulation.cache.iceflow, simulation, glacier_idx, t, θ)
        Vx, Vy = Huginn.surface_V(H, simulation, t, θ)
        return sum(Vx.*inn1(vecBackwardSIA2D[1])+Vy.*inn1(vecBackwardSIA2D[2]))
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
            use_velocities=true,
            tspan=tspan,
            multiprocessing=false,
            test_mode=true,
            rgi_paths=rgi_paths,
            gridScalingFactor=4),
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
            progress=true)
    )

    nn_model = NeuralNetwork(params)

    model = if target==:A
        iceflow_model = SIA2Dmodel(params; A=LawA(nn_model, params; precompute_VJPs=false))
        Model(
            iceflow = iceflow_model,
            mass_balance = nothing,
            regressors = (; A=nn_model),
        )
    else
        throw("Unsupported target $(target)")
    end

    glaciers = initialize_glaciers(rgi_ids, params)

    glacier_idx = 1

    H = glaciers[glacier_idx].H₀
    simulation = Inversion(model, glaciers, params)

    t = tspan[1]
    θ = simulation.model.trainable_components.θ
    cache = init_cache(model, simulation, glacier_idx, θ)
    simulation.cache = cache

    vecBackwardSIA2D = [
        randn(size(H, 1), size(H, 2)),
        randn(size(H, 1), size(H, 2))]

    apply_all_callback_laws!(simulation.model.iceflow, simulation.cache.iceflow, simulation, glacier_idx, t, θ)
    Vx, Vy = Huginn.surface_V(H, simulation, t, θ)

    ∂H, = ODINN.VJP_λ_∂surface_V∂H(
        adjointFlavor.VJP_method,
        vecBackwardSIA2D[1],
        vecBackwardSIA2D[2],
        H,
        θ,
        simulation,
        t,
    )
    ∂θ, = ODINN.VJP_λ_∂surface_V∂θ(
        adjointFlavor.VJP_method,
        vecBackwardSIA2D[1],
        vecBackwardSIA2D[2],
        H,
        θ,
        simulation,
        t,
    )

    # Check gradient wrt H
    function f_H(H, args)
        simulation, t, vecBackwardSIA2D, glacier_idx = args
        θ = simulation.model.trainable_components.θ
        return _loss(H, θ, simulation, t, vecBackwardSIA2D, glacier_idx)
    end
    ratio = []
    angle = []
    relerr = []
    eps = []
    for k in range(3,8)
        ϵ = 10.0^(-k)
        push!(eps, ϵ)
        ∂H_num = compute_numerical_gradient(H, (simulation, t, vecBackwardSIA2D, glacier_idx), f_H, ϵ; varStr="of H")
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
        H, simulation, t, vecBackwardSIA2D, glacier_idx = args
        return _loss(H, θ, simulation, t, vecBackwardSIA2D, glacier_idx)
    end
    ratio = []
    angle = []
    relerr = []
    eps = []
    for k in range(5,7)
        ϵ = 10.0^(-k)
        push!(eps, ϵ)
        ∂θ_num = compute_numerical_gradient(θ, (H, simulation, t, vecBackwardSIA2D, glacier_idx), f_θ, ϵ; varStr="of θ")
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
