using Distributed: map

function test_grad_finite_diff(
    adjointFlavor::ADJ;
    thres = [0., 0., 0.],
    target = :A,
    finite_difference_method = :FiniteDifferences,
    finite_difference_order = 3,
    loss = LossH(),
    train_initial_conditions = false,
    multiglacier = false,
    use_MB = false,
    custom_NN = false,
    max_params = 40,
) where {ADJ<:AbstractAdjointMethod}

    print("> Testing target $(target) with adjoint $(adjointFlavor) and loss $(Base.typename(typeof(loss)).name)")
    println(use_MB ? " and with MB" : "")

    # Determine if we are working with a velocity loss
    velocityLoss = typeof(loss) <: Union{<: LossV, <: LossHV}

    thres_ratio = thres[1]
    thres_angle = thres[2]
    thres_relerr = thres[3]

    rgi_ids = @match (velocityLoss, multiglacier) begin
        (true, false) => ["RGI60-11.03646"]
        (false, false) => ["RGI60-11.03638"]
        (false, true) => ["RGI60-11.03638", "RGI60-11.01450"]
    end

    rgi_paths = get_rgi_paths()

    working_dir = joinpath(ODINN.root_dir, "test/data")

    δt = 1/12
    tspan = use_MB ? (1980.0, 2019.0) : (2010.0, 2012.0)

    if isa(adjointFlavor, ODINN.SciMLSensitivityAdjoint)
        optim_autoAD = Optimization.AutoEnzyme()
        sensealg = GaussAdjoint(autojacvec=SciMLSensitivity.EnzymeVJP())
    else
        optim_autoAD = ODINN.NoAD()
        sensealg = SciMLSensitivity.ZygoteAdjoint()
    end

    params = Parameters(
        simulation = SimulationParameters(
            working_dir = working_dir,
            use_MB = use_MB,
            use_velocities = true,
            tspan = tspan,
            step = δt,
            multiprocessing = false,
            workers = 1,
            test_mode = true,
            rgi_paths = rgi_paths,
            gridScalingFactor = 4
            ),
        hyper = Hyperparameters(
            batch_size = length(rgi_ids), # We set batch size equals all datasize so we test gradient
            epochs = 100,
            optimizer = ODINN.ADAM(0.005)
            ),
        physical = PhysicalParameters(
            # When MB is being tested, reduce the impact of creeping so that the gradient is dominated by the MB contribution
            minA = use_MB ? 1e-21 : 8e-21,
            maxA = use_MB ? 2e-21 : 8e-18
            ),
        UDE = UDEparameters(
            sensealg = sensealg,
            optim_autoAD = optim_autoAD,
            grad = adjointFlavor,
            optimization_method = "AD+AD",
            empirical_loss_function = loss,
            target = target,
            initial_condition_filter = :Zang1980
            ),
        solver = Huginn.SolverParameters(
            step = δt,
            save_everystep = true,
            progress = true
            )
    )

    # We retrieve some glaciers for the simulation
    # Time stanpshots for transient inversion
    tstops = collect(tspan[1]:δt:tspan[2])
    kwargs = velocityLoss ? (;
        velocityDatacubes = Dict(
            rgi_ids[1] => Sleipnir.fake_multi_datacube()
        )
    ) : NamedTuple()
    model = Model(
        iceflow = SIA2Dmodel(params; A = ConstantA(2.21e-18)),
        mass_balance = TImodel1(params; DDF = 6.0/1000.0, acc_factor = 1.2/1000.0),
    )
    glaciers = initialize_glaciers(rgi_ids, params; kwargs...)
    glaciers = generate_ground_truth(glaciers, params, model, tstops)

    # Neural network model
    if custom_NN
        function normalize(v::Union{Vector,SubArray})
            @assert length(v) == 2
            return [ODINN.normalize(v[1]; lims = (0.0, 200.0)), ODINN.normalize(v[2]; lims = (0.0, 0.6))]
        end
        function scale(v::Union{Vector,SubArray})
            @assert length(v) == 1
            return 1e2 .* v
        end
        architecture = Lux.Chain(
            Lux.WrappedFunction(x -> LuxFunction(normalize, x)),
            Lux.Dense(2, 5, x -> gelu.(x)),
            Lux.Dense(5, 10, x -> gelu.(x)),
            Lux.Dense(10, 5, x -> gelu.(x)),
            Lux.Dense(5, 1, sigmoid),
            Lux.WrappedFunction(x -> LuxFunction(scale, x))
        )
        nn_model = NeuralNetwork(params; architecture = architecture)
    else
        nn_model = NeuralNetwork(params)
    end

    ic = if train_initial_conditions
        InitialCondition(params, glaciers, :Farinotti2019)
    else
        nothing
    end

    # Define regressors for each test
    regressors = @match (target, train_initial_conditions) begin
        (:A, false) => (; A = nn_model)
        (:A, true) => (; A = nn_model, IC = ic)
        (:D_hybrid, false) =>  (; Y = nn_model)
        (:D_hybrid, true) => (; Y = nn_model, IC = ic)
        (:D, false) => (; U = nn_model)
        (:D, true) => (; U = nn_model, IC = ic)
    end

    model = @match target begin
        :A => Model(
            iceflow = SIA2Dmodel(params; A = LawA(nn_model, params)),
            mass_balance = TImodel1(params; DDF = 6.0/1000.0, acc_factor = 1.2/1000.0),
            regressors = regressors,
        )
        :D_hybrid => Model(
            iceflow = SIA2Dmodel(params; Y = LawY(nn_model, params)),
            mass_balance = TImodel1(params; DDF = 6.0/1000.0, acc_factor= 1.2/1000.0),
            regressors = regressors,
        )
        :D => Model(
            iceflow = SIA2Dmodel(params; U = LawU(nn_model, params)),
            mass_balance = TImodel1(params; DDF = 6.0/1000.0, acc_factor = 1.2/1000.0),
            regressors = regressors,
            target = SIA2D_D_target(
                interpolation = :Linear,
                n_interp_half = 200,
            ),
        )
    end

    # We create an ODINN prediction
    functional_inversion = FunctionalInversion(model, glaciers, params)
    simulation = functional_inversion

    θ = simulation.model.machine_learning.θ
    n_params = length(θ)

    glacier_idx = 1
    simulation.cache = init_cache(model, simulation, glacier_idx, θ)

    loss_iceflow_grad!(dθ, _θ, _simulation) = if isa(adjointFlavor, ODINN.SciMLSensitivityAdjoint)
        ret = ODINN.grad_loss_iceflow!(_θ, simulation, map)
        @assert !any(isnan, ret) "Gradient computed with SciML contains NaNs. Try to run the code again if you just started the REPL. Gradient is $(ret)"
        dθ .= ret
    else
        SIA2D_grad!(dθ, _θ, _simulation)
    end

    function f(x, simulation)
        simulation.model.machine_learning.θ = x
        return ODINN.loss_iceflow_transient(x, simulation, map)
    end

    dθ = zero(θ)
    if !isa(adjointFlavor, ODINN.SciMLSensitivityAdjoint)
        loss_iceflow_grad!(dθ, θ, simulation)
    else
        # Computation of the gradient with SciMLSensitivity can fail with a fresh REPL
        # Running the same code a second or third time usually works
        # More information in https://github.com/ODINN-SciML/ODINN.jl/issues/354
        try
            loss_iceflow_grad!(dθ, θ, simulation)
        catch
            @warn "Computation of gradient with SciMLSensitivity fail with first run due to compilation errors. Trying for a second time..."
            try
                loss_iceflow_grad!(dθ, θ, simulation)
                @warn "Computation of gradient with SciMLSensitivity succeded in a second run after compilation."
            catch
                @warn "Computation of gradient with SciMLSensitivity fail after second run due to compilation errors. Trying one last time..."
                loss_iceflow_grad!(dθ, θ, simulation)
            end
        end
    end
    JET.@test_opt broken=true target_modules=(Sleipnir, Muninn, Huginn, ODINN) loss_iceflow_grad!(dθ, θ, simulation)
    JET.@test_opt broken=true target_modules=(Sleipnir, Muninn, Huginn, ODINN) ODINN.loss_iceflow_transient(θ, simulation, map)

    if finite_difference_method == :FiniteDifferences

        ### Further computes derivatives with FiniteDifferences.jl (stepsize algorithm included)

        if n_params > max_params
            # Evaluate gradient on subset of parameters
            # We just evaluate in a subset to save some computation
            @info "Testing gradient with a subset of parameters since the original parameter vector θ is of dimension $(n_params)."

            # Component array with binary entry
            θ_mask = θ .== nothing

            for key in keys(θ)
                if key == :IC
                    # Initial condition
                    for glacier in glaciers
                        M = ODINN.evaluate_H₀(θ, glacier, params.UDE.initial_condition_filter)
                        non_zero = M .> 1.0
                        idxs = rand(findall(non_zero), max_params)
                        mask = falses(size(M)...)
                        mask[idxs] .= 1
                        key = Symbol("$(glacier.rgi_id)")
                        θ_mask.IC[key] .= mask
                    end
                else
                    # Mask parameter vector
                    indx = ODINN.sample(1:length(θ.U), max_params; replace = false)
                    view(θ_mask, key)[indx] .= true
                end
            end

            function f_subset(x, simulation, mask)
                α = copy(θ)
                α[mask] = x
                return f(α, simulation)
            end

            dθ_FD, = FiniteDifferences.grad(
                central_fdm(finite_difference_order, 1),
                α -> f_subset(α, simulation, θ_mask),
                θ[θ_mask]
            )
            dθ = dθ[θ_mask]
        else
            # Compute gradient with all parameters
            dθ_FD, = FiniteDifferences.grad(
                central_fdm(finite_difference_order, 1),
                _θ -> f(_θ, simulation),
                θ
            )
        end

        ratio_FD, angle_FD, relerr_FD = stats_err_arrays(dθ, dθ_FD)
        printVecScientific("ratio  = ", [ratio_FD], thres_ratio)
        printVecScientific("angle  = ", [angle_FD], thres_angle)
        printVecScientific("relerr = ", [relerr_FD], thres_relerr)
        @test abs(ratio_FD) < thres_ratio
        @test abs(angle_FD) < thres_angle
        @test abs(relerr_FD) < thres_relerr

    elseif finite_difference_method == :Manual

        ### Manual finite differences with different choices of stepsize

        ratio = []
        angle = []
        relerr = []
        eps = []
        for k in range(3, 8)
            ϵ = 10.0^(-k)
            push!(eps, ϵ)
            dθ_num = compute_numerical_gradient(θ, (simulation), f, ϵ; varStr="of θ")
            ratio_k, angle_k, relerr_k = stats_err_arrays(dθ, dθ_num)
            push!(ratio, ratio_k)
            push!(angle, angle_k)
            push!(relerr, relerr_k)
        end
        min_ratio = minimum(abs.(ratio))
        min_angle = minimum(abs.(angle))
        min_relerr = minimum(abs.(relerr))

        if printDebug | !( (min_ratio < thres_ratio) & (min_angle < thres_angle) & (min_relerr < thres_relerr) )
            println("eps    = ",printVecScientific(eps))
            printVecScientific("ratio  = ", ratio, thres_ratio)
            printVecScientific("angle  = ", angle, thres_angle)
            printVecScientific("relerr = ", relerr, thres_relerr)
        end
        @test min_ratio < thres_ratio
        @test min_angle < thres_angle
        @test min_relerr < thres_relerr

    else
        throw("Finite difference method not implemented.")
    end

end

function test_grad_loss_term()
    function _loss!(l, a, b, norm, lossType)
        l[1] = loss(lossType, a, b; normalization=norm)
        return nothing
    end

    lossType = L2Sum(distance=2)
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
    if printDebug | !( (abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres) )
        printVecScientific("ratio  = ",[ratio],thres)
        printVecScientific("angle  = ",[angle],thres)
        printVecScientific("relerr = ",[relerr],thres)
    end
    @test (abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres)
end

function _loss_halfar!(l, R₀, h₀, r₀, A, n, tstops, H_ref, params, lossType, glacier, θ)
    normalization = 1.0
    l_H = 0.0
    Δt = diff(tstops)
    physicalParams = params.physical
    for τ in range(2,length(tstops))
        t₁ = tstops[τ]
        _H₁ = halfar_solution(R₀, t₁, h₀, r₀, A[1], n, physicalParams)
        mean_error, = loss(
            lossType,
            _H₁,
            H_ref[τ],
            t=t₁,
            glacier=glacier,
            θ=θ,
            params=params;
            normalization=prod(size(H_ref[τ]))/normalization
        )
        l_H += Δt[τ-1] * mean_error
    end
    l[1] = l_H
    return nothing
end

function test_grad_Halfar(
    adjointFlavor::ADJ;
    thres=[0., 0., 0.]
    ) where {ADJ <: AbstractAdjointMethod}

    lossType = LossH(L2Sum(distance=15))
    A = 8e-19
    t₀ = 5.0
    t₁ = 30.0
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

    # Get parameters for a simulation
    parameters = Parameters(
        simulation=SimulationParameters(
            tspan=(t₀, t₁),
            step=δt,
            multiprocessing=false,
            use_MB=false,
            use_iceflow=true,
            test_mode=true,
            working_dir=Huginn.root_dir
        ),
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-18),
        UDE = UDEparameters(
            optim_autoAD=ODINN.NoAD(),
            grad=adjointFlavor,
            optimization_method="AD+AD",
            empirical_loss_function=lossType,
            target = :A
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

    # Use a constant A for testing
    model = Model(
        iceflow = SIA2Dmodel(parameters),#; A=A_law),
        mass_balance = nothing,
        machine_learning = NeuralNetwork(parameters)
    )

    θ = model.machine_learning.θ
    modelNN = model.machine_learning.architecture
    st = model.machine_learning.st
    smodel = StatefulLuxLayer{true}(modelNN, θ.θ, st)
    min_NN = parameters.physical.minA
    max_NN = parameters.physical.maxA
    A_θ = ODINN.predict_A̅(smodel, [T], (min_NN, max_NN))[1]
    println("A_θ = ",A_θ)

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
    # TODO: add law
    glaciers = generate_ground_truth(glaciers, parameters, model, tstops)

    model.iceflow = SIA2Dmodel(parameters)

    # We create an ODINN prediction
    simulation = FunctionalInversion(model, glaciers, parameters)

    # Compute gradient of with Halfar solution wrt A
    A_θ = [A_θ]
    ∂A_enzyme = Enzyme.make_zero(A_θ)
    dl_enzyme = [1.]
    l_enzyme = Enzyme.make_zero(dl_enzyme)
    H_ref = simulation.glaciers[1].thicknessData.H
    Enzyme.autodiff(
        Reverse, _loss_halfar!, Const,
        Duplicated(l_enzyme, dl_enzyme),
        Enzyme.Const(R₀),
        Enzyme.Const(h₀),
        Enzyme.Const(r₀),
        Duplicated(A_θ, ∂A_enzyme),
        Enzyme.Const(n),
        Enzyme.Const(tstops),
        Enzyme.Const(H_ref),
        Enzyme.Const(parameters),
        Enzyme.Const(lossType),
        Enzyme.Const(glacier),
        Enzyme.Const(θ),
    )

    # _loss_halfar!(l_enzyme, R₀, h₀, r₀, A_θ, n, tstops, H_ref, physicalParams, lossType)
    println("l_enzyme=", l_enzyme)
    println("∂A_enzyme=", ∂A_enzyme)

    # Retrieve apply parametrization from inversion
    # TODO: replace function below
    ∇θ, = Zygote.gradient(_θ -> apply_parametrization(
        model.machine_learning.target;
        H = nothing, ∇S = nothing, θ = _θ,
        iceflow_model = only(model.iceflow), ml_model = model.machine_learning,
        glacier = only(glaciers), params = parameters),
        θ)
    dθ_halfar = ∂A_enzyme[1] * ∇θ

    # Compute gradient with manual implementation of the backward + discrete adjoint of SIA2D
    dθ = zero(θ)
    SIA2D_grad!(dθ, θ, simulation)

    ratio, angle, relerr = stats_err_arrays(dθ, dθ_halfar)

    # TODO: fix this test
    thres_ratio = thres[1]
    thres_angle = thres[2]
    thres_relerr = thres[3]
    if printDebug | !( (abs(ratio)<thres_ratio) & (abs(angle)<thres_angle) & (abs(relerr)<thres_relerr) )
        printVecScientific("ratio  = ", [ratio], thres_ratio)
        printVecScientific("angle  = ", [angle], thres_angle)
        printVecScientific("relerr = ", [relerr], thres_relerr)
    end
    @test abs(ratio) < thres_ratio
    @test abs(angle) < thres_angle
    @test abs(relerr) < thres_relerr
end
