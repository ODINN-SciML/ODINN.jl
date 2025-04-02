
function test_grad_discreteAdjoint_discreteVJP()

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
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-18),
        UDE = UDEparameters(
            sensealg=SciMLSensitivity.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=DiscreteAdjoint(VJP_method=DiscreteVJP()),
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
        ratio_k, angle_k, relerr_k = stats_err_arrays(dθ, dθ_num)
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
    @test min_ratio<thres_ratio
    @test min_angle<thres_angle
    @test min_relerr<thres_relerr

end


function test_grad_discreteAdjoint_continuousVJP()

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
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-18),
        UDE = UDEparameters(
            sensealg=SciMLSensitivity.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=DiscreteAdjoint(VJP_method=ContinuousVJP()),
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
        ratio_k, angle_k, relerr_k = stats_err_arrays(dθ, dθ_num)
        push!(ratio, ratio_k)
        push!(angle, angle_k)
        push!(relerr, relerr_k)
    end
    min_ratio = minimum(abs.(ratio))
    min_angle = minimum(abs.(angle))
    min_relerr = minimum(abs.(relerr))
    thres_ratio = 1e-2
    thres_angle = 1e-8
    thres_relerr = 3e-2
    if !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
        println("eps    = ",printVecScientific(eps))
        println("ratio  = ",printVecScientific(ratio))
        println("angle  = ",printVecScientific(angle))
        println("relerr = ",printVecScientific(relerr))
    end
    @test min_ratio<thres_ratio
    @test min_angle<thres_angle
    @test min_relerr<thres_relerr


end

function test_grad_continuousAdjoint_discreteVJP()

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
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-18),
        UDE = UDEparameters(
            sensealg=SciMLSensitivity.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=ContinuousAdjoint(VJP_method=DiscreteVJP()),
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
        ratio_k, angle_k, relerr_k = stats_err_arrays(dθ, dθ_num)
        push!(ratio, ratio_k)
        push!(angle, angle_k)
        push!(relerr, relerr_k)
    end
    min_ratio = minimum(abs.(ratio))
    min_angle = minimum(abs.(angle))
    min_relerr = minimum(abs.(relerr))
    thres_ratio = 1e-2
    thres_angle = 1e-8
    thres_relerr = 3e-2
    # if !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
        println("eps    = ",printVecScientific(eps))
        println("ratio  = ",printVecScientific(ratio))
        println("angle  = ",printVecScientific(angle))
        println("relerr = ",printVecScientific(relerr))
    # end
    @test min_ratio<thres_ratio
    @test min_angle<thres_angle
    @test min_relerr<thres_relerr

end

function test_grad_loss_term()
    function _loss!(l, a, b, norm, lossType)
        l[1] = loss(lossType, a, b; normalization=norm)
        return nothing
    end


    lossType = L2Sum()
    nx = 4
    ny = 5
    norm = 3.5
    a = randn(nx, ny)
    b = randn(nx, ny)
    l = [0.]
    _loss!(l, a, b, norm, lossType)
    dl_enzyme = [1.]
    l_enzyme = Enzyme.make_zero(dl_enzyme)
    da_enzyme = Enzyme.make_zero(a)
    Enzyme.autodiff(
        Reverse, _loss!, Const,
        Duplicated(l_enzyme, dl_enzyme),
        Duplicated(a, da_enzyme),
        Enzyme.Const(b),
        Enzyme.Const(norm),
        Enzyme.Const(lossType),
    )
    da = backward_loss(lossType, a, b; normalization=norm)
    ratio, angle, relerr = stats_err_arrays(da, da_enzyme)
    thres = 1e-14
    if !( (abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres) )
        println("ratio  = ",ratio)
        println("angle  = ",angle)
        println("relerr = ",relerr)
    end
    @test (abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres)


    lossType = L2SumWithinGlacier(distance=2)
    nx = 9
    ny = 10
    norm = 3.5
    a = randn(nx, ny)
    a[1,:] .= 0; a[end,:] .= 0; a[:,1] .= 0; a[:,end] .= 0
    b = randn(nx, ny)
    b[1,:] .= 0; b[end,:] .= 0; b[:,1] .= 0; b[:,end] .= 0
    l = [0.]
    _loss!(l, a, b, norm, lossType)
    dl_enzyme = [1.]
    l_enzyme = Enzyme.make_zero(dl_enzyme)
    da_enzyme = Enzyme.make_zero(a)
    Enzyme.autodiff(
        set_runtime_activity(Reverse), _loss!, Const,
        Duplicated(l_enzyme, dl_enzyme),
        Duplicated(a, da_enzyme),
        Enzyme.Const(b),
        Enzyme.Const(norm),
        Enzyme.Const(lossType),
    )
    da = backward_loss(lossType, a, b; normalization=norm)
    ratio, angle, relerr = stats_err_arrays(da, da_enzyme)
    thres = 1e-14
    if !( (abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres) )
        println("ratio  = ",ratio)
        println("angle  = ",angle)
        println("relerr = ",relerr)
    end
    @test (abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres)
end

function test_grad_discreteAdjoint_Halfar()
    Random.seed!(1234)

    function _loss!(l, R₀, h₀, r₀, A, n, tstops, H_ref, physicalParams, lossType)
        normalization = 1.0
        l_H = 0.0
        Δt = diff(tstops)
        for τ in range(2,length(tstops))
            t₁ = tstops[τ]
            _H₁ = halfar_solution(R₀, t₁, h₀, r₀, A[1], n, physicalParams)
            mean_error = loss(lossType, _H₁, H_ref[τ]; normalization=prod(size(H_ref[τ]))/normalization)
            l_H += Δt[τ-1] * mean_error
        end
        l[1] = l_H
        return nothing
    end

    lossType = L2SumWithinGlacier()
    A = 8e-19
    t₀ = 5.0
    t₁ = 6.0
    h₀ = 500
    r₀ = 1000
    n = 3.0
    Δx = 50.0
    Δy = 50.0
    nx = 100
    ny = 100
    δt = 1/12
    T = 2.0
    tstops = collect(t₀:δt:t₁)
    println("tstops=",tstops)

    # Get parameters for a simulation
    parameters = Parameters(
        simulation=SimulationParameters(
            tspan=(t₀, t₁),
            step=δt,
            multiprocessing=false,
            use_MB=false,
            use_iceflow=true,
            light=false, # for now we do the simulation like this (a better name would be dense)
            test_mode=true,
            working_dir=Huginn.root_dir
        ),
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-18),
        UDE = UDEparameters(
            optim_autoAD=ODINN.NoAD(),
            grad=DiscreteAdjoint(VJP_method=DiscreteVJP()),
            optimization_method="AD+AD",
            empirical_loss_function=lossType,
            target = "A"
        ),
        solver=SolverParameters(
            reltol=1e-12,
            step=δt,
            save_everystep=true,
            progress=true
        )
    )

    # Bed (it has to be flat for the Halfar solution)
    B = zeros((nx,ny))
    model = Model(iceflow = SIA2Dmodel(parameters), mass_balance = nothing, machine_learning = NeuralNetwork(parameters))

    θ = model.machine_learning.θ
    modelNN = model.machine_learning.architecture
    st = model.machine_learning.st
    smodel = StatefulLuxLayer{true}(modelNN, θ.θ, st)
    min_NN = parameters.physical.minA
    max_NN = parameters.physical.maxA
    A_θ = ODINN.predict_A̅(smodel, [T], (min_NN, max_NN))[1]
    println("A_θ=",A_θ)

    # Initial condition of the glacier
    R₀ = [sqrt((Δx * (i - nx/2))^2 + (Δy * (j - ny/2))^2) for i in 1:nx, j in 1:ny]
    H₀ = halfar_solution(R₀, t₀, h₀, r₀, A_θ, n, parameters.physical)
    S = B + H₀

    # Define glacier object
    climate = Sleipnir.DummyClimate2D(longterm_temps=[T])
    glacier = Glacier2D(rgi_id = "toy", climate = climate, H₀ = H₀, S = S, B = B, A = A, n=n,
                        Δx=Δx, Δy=Δy, nx=nx, ny=ny, C = 0.0)
    glaciers = Vector{Sleipnir.AbstractGlacier}([glacier])

    fakeA(T) = A
    map(glacier -> ODINN.generate_ground_truth(glacier, fakeA, parameters, model, tstops), glaciers)
    # TODO: This function messes up with the model variable, for now we do a clean restart
    model.iceflow = SIA2Dmodel(parameters)

    # We create an ODINN prediction
    simulation = FunctionalInversion(model, glaciers, parameters)
    physicalParams = parameters.physical

    # Compute gradient of with Halfar solution wrt A
    A_θ = [A_θ]
    ∂A_enzyme = Enzyme.make_zero(A_θ)
    dl_enzyme = [1.]
    l_enzyme = Enzyme.make_zero(dl_enzyme)
    H_ref = only(simulation.glaciers[1].data).H
    Enzyme.autodiff(
        Reverse, _loss!, Const,
        Duplicated(l_enzyme, dl_enzyme),
        Enzyme.Const(R₀),
        Enzyme.Const(h₀),
        Enzyme.Const(r₀),
        Duplicated(A_θ, ∂A_enzyme),
        Enzyme.Const(n),
        Enzyme.Const(tstops),
        Enzyme.Const(H_ref),
        Enzyme.Const(physicalParams),
        Enzyme.Const(lossType),
    )

    # _loss!(l_enzyme, R₀, h₀, r₀, A_θ, n, tstops, H_ref, physicalParams, lossType)
    println("l_enzyme=",l_enzyme)
    println("∂A_enzyme=",∂A_enzyme)

    ∇θ, = Zygote.gradient(_θ -> ODINN.grad_apply_UDE_parametrization(_θ, simulation, 1), θ)
    # println("∇θ in test=",∇θ)
    dθ_halfar = ∂A_enzyme[1]*∇θ

    # Compute gradient with manual implementation of the backward + discrete adjoint of SIA2D
    dθ = zero(θ)
    SIA2D_grad!(dθ, θ, simulation)

    ratio, angle, relerr = stats_err_arrays(dθ, dθ_halfar)

    # TODO: fix this test
    thres_ratio = 1e0
    thres_angle = 1e-15
    thres_relerr = 1e1
    if !( (abs(ratio)<thres_ratio) & (abs(angle)<thres_angle) & (abs(relerr)<thres_relerr) )
        println("ratio  = ",ratio)
        println("angle  = ",angle)
        println("relerr = ",relerr)
    end
    @test (abs(ratio)<thres_ratio) & (abs(angle)<thres_angle) & (abs(relerr)<thres_relerr)
end
