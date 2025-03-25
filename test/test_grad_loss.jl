
function test_grad_discreteAdjoint()

    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()

    working_dir = joinpath(ODINN.root_dir, "test/data")

    δt = 1/12

    params = Parameters(
        simulation = SimulationParameters(
            working_dir=working_dir,
            use_MB=false,
            velocities=true,
            tspan=(2010.0, 2015.0),
            step=δt,
            multiprocessing=false,
            workers=1,
            light=false, # for now we do the simulation like this (a better name would be dense)
            test_mode=true,
            rgi_paths=rgi_paths),
        hyper = Hyperparameters(
            batch_size=length(rgi_ids), # We set batch size equals all datasize so we test gradient
            epochs=100,
            optimizer=ODINN.ADAM(0.005)),
            # optimizer=ODINN.Descent(0.001)),
        UDE = UDEparameters(
            sensealg=ODINN.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=ODINNDiscreteAdjoint(),
            optimization_method="AD+AD",
            target = "A"),
        solver = Huginn.SolverParameters(
            step=δt,
            save_everystep=true,
            progress=true)
    )

    model = Model(
        iceflow = SIA2Dmodel(params),
        mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
        machine_learning = NeuralNetwork(params))

    # We retrieve some glaciers for the simulation
    glaciers = initialize_glaciers(rgi_ids, params)

    # Time stanpshots for transient inversion
    tstops = collect(2010:δt:2015)

    # Overwrite constant A fake function for testing
    fakeA(T) = 2.21e-18

    map(glacier -> ODINN.generate_ground_truth(glacier, fakeA, params, model, tstops), glaciers)
    # TODO: This function does shit on the model variable, for now we do a clean restart
    model.iceflow = SIA2Dmodel(params)

    # We create an ODINN prediction
    functional_inversion = FunctionalInversion(model, glaciers, params)
    simulation = functional_inversion


    θ = simulation.model.machine_learning.θ
    loss_function(_θ, (_simulation)) = ODINN.loss_iceflow_transient(_θ, _simulation)
    loss_iceflow_grad!(dθ, θ, _simulation) = SIA2D_grad!(dθ, θ, _simulation)

    function f(x, simulation)
        return loss_function(x, simulation)
    end

    dθ=zero(θ)
    loss_iceflow_grad!(dθ, θ, simulation)

    ratio = []
    angle = []
    relerr = []
    eps = []
    for k in range(3,8)
        ϵ = 10.0^(-k)
        push!(eps, ϵ)
        dθ_num = compute_numerical_gradient(θ, (simulation), f, ϵ)
        ratio_k, angle_k, relerr_k = stats_err_backward(dθ, dθ_num)
        push!(ratio, ratio_k)
        push!(angle, angle_k)
        push!(relerr, relerr_k)
    end
    min_ratio = minimum(abs.(ratio))
    min_angle = minimum(abs.(angle))
    min_relerr = minimum(abs.(relerr))
    thres_ratio = 1e-2
    thres_angle = 1e-8
    thres_relerr = 1e-2
    if !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
        println("eps    = ",printVecScientific(eps))
        println("ratio  = ",printVecScientific(ratio))
        println("angle  = ",printVecScientific(angle))
        println("relerr = ",printVecScientific(relerr))
    end
    @test (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr)

end


function test_grad_continuousAdjoint()

    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()

    working_dir = joinpath(ODINN.root_dir, "test/data")

    δt = 1/12

    params = Parameters(
        simulation = SimulationParameters(
            working_dir=working_dir,
            use_MB=false,
            velocities=true,
            tspan=(2010.0, 2015.0),
            step=δt,
            multiprocessing=false,
            workers=1,
            light=false, # for now we do the simulation like this (a better name would be dense)
            test_mode=true,
            rgi_paths=rgi_paths),
        hyper = Hyperparameters(
            batch_size=length(rgi_ids), # We set batch size equals all datasize so we test gradient
            epochs=100,
            optimizer=ODINN.ADAM(0.005)),
            # optimizer=ODINN.Descent(0.001)),
        UDE = UDEparameters(
            sensealg=ODINN.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=ODINNContinuousAdjoint(),
            optimization_method="AD+AD",
            target = "A"),
        solver = Huginn.SolverParameters(
            step=δt,
            save_everystep=true,
            progress=true)
    )

    model = Model(
        iceflow = SIA2Dmodel(params),
        mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
        machine_learning = NeuralNetwork(params))

    # We retrieve some glaciers for the simulation
    glaciers = initialize_glaciers(rgi_ids, params)

    # Time stanpshots for transient inversion
    tstops = collect(2010:δt:2015)

    # Overwrite constant A fake function for testing
    fakeA(T) = 2.21e-18

    map(glacier -> ODINN.generate_ground_truth(glacier, fakeA, params, model, tstops), glaciers)
    # TODO: This function does shit on the model variable, for now we do a clean restart
    model.iceflow = SIA2Dmodel(params)

    # We create an ODINN prediction
    functional_inversion = FunctionalInversion(model, glaciers, params)
    simulation = functional_inversion


    θ = simulation.model.machine_learning.θ
    loss_function(_θ, (_simulation)) = ODINN.loss_iceflow_transient(_θ, _simulation)
    loss_iceflow_grad!(dθ, θ, _simulation) = SIA2D_grad!(dθ, θ, _simulation)

    function f(x, simulation)
        return loss_function(x, simulation)
    end

    dθ=zero(θ)
    loss_iceflow_grad!(dθ, θ, simulation)

    ratio = []
    angle = []
    relerr = []
    eps = []
    for k in range(3,8)
        ϵ = 10.0^(-k)
        push!(eps, ϵ)
        dθ_num = compute_numerical_gradient(θ, (simulation), f, ϵ)
        ratio_k, angle_k, relerr_k = stats_err_backward(dθ, dθ_num)
        push!(ratio, ratio_k)
        push!(angle, angle_k)
        push!(relerr, relerr_k)
    end
    min_ratio = minimum(abs.(ratio))
    min_angle = minimum(abs.(angle))
    min_relerr = minimum(abs.(relerr))
    thres_ratio = 1e-6
    thres_angle = 1e-6
    thres_relerr = 1e-6
    if !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
        println("eps    = ",printVecScientific(eps))
        println("ratio  = ",printVecScientific(ratio))
        println("angle  = ",printVecScientific(angle))
        println("relerr = ",printVecScientific(relerr))
    end
    @test (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr)

end
