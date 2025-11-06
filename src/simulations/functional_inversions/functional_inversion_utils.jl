import Huginn.precompute_all_VJPs_laws!, Huginn.run!

"""
    run!(simulation::FunctionalInversion)

Run the training process for a given `FunctionalInversion` simulation.

# Arguments
- `simulation::FunctionalInversion`: The simulation object containing the parameters and settings for the functional inversion process.

# Description
This function initiates the training of a Universal Differential Equation (UDE) for the provided simulation. It prints a message indicating the start of the training process, calls the `train_UDE!` function to perform the training, and collects the results in `results_list`. The results are intended to be saved using `Sleipnir.save_results_file!`, but this step is currently commented out and will be enabled once the optimization is working. Finally, the garbage collector is triggered to free up memory.

# Notes
- The `Sleipnir.save_results_file!` function call is currently commented out and should be enabled once the optimization process is confirmed to be working.
- The garbage collector is explicitly run using `GC.gc()` to manage memory usage.
"""
function run!(
    simulation::FunctionalInversion;
    path::Union{String, Nothing} = nothing,
    file_name::Union{String, Nothing} = nothing,
    save_every_iter::Bool = false,
    path_tb_logger::Union{String, Nothing} = joinpath(
        ODINN.root_dir,
        ".log/",
        Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"),
    ),
)
    println("Training UDE...\n")

    # Set expected total number of epochs from beginning for the callback
    simulation.results.stats.niter = sum(simulation.parameters.hyper.epochs)

    logger = isnothing(path_tb_logger) || simulation.parameters.simulation.test_mode ? nothing : TBLogger(path_tb_logger)
    if !(typeof(simulation.parameters.hyper.optimizer) <: Vector)
        # One single optimizer
        sol = train_UDE!(simulation; save_every_iter=save_every_iter, logger=logger)
    else
        # Multiple optimizers
        optimizers = simulation.parameters.hyper.optimizer
        epochs = simulation.parameters.hyper.epochs
        @assert length(optimizers) == length(epochs) "Provide number of epochs as a vector with the same length of optimizers"
        for i in 1:length(epochs)
            # Construct a new simulation for each optimizer
            simulation.parameters.hyper.optimizer = optimizers[i]
            simulation.parameters.hyper.epochs = epochs[i]
            if i !== 1
                θ_trained = sol.u
                simulation.model.machine_learning.θ = θ_trained
            end
            sol = train_UDE!(simulation; save_every_iter=save_every_iter, logger=logger)
            # Clear results of previous simulation for fresh start
            if i < length(epochs)
                simulation.results.simulation = Sleipnir.Results{Float64, Int64}[]
            end
        end
    end

    # Setup final results
    simulation.model.machine_learning.θ = sol.u

    simulation.results.stats.niter = length(simulation.results.stats.losses)
    # Final parameters of the neural network for the target regressor
    # Just one neural network is supported for now
    reg_key = only(intersect(keys(sol.u), (:A, :C, :n, :Y, :U)))
    simulation.results.stats.θ = sol.u[reg_key]
    # Final initial conditions for the simulation
    if haskey(sol.u, :IC)
        simulation.results.stats.initial_conditions = Dict()
        for glacier in simulation.glaciers
            simulation.results.stats.initial_conditions[String(glacier.rgi_id)] = evaluate_H₀(
                simulation.model.machine_learning.θ,
                glacier,
                simulation.parameters.UDE.initial_condition_filter
            )
        end
    end

    # TODO: Save when optimization is working
    # Save results in path is provided
    if !isnothing(path) & !isnothing(file_name)
        ODINN.save_inversion_file!(sol, simulation; path = path, file_name = file_name)
    end

    @everywhere GC.gc() # run garbage collector

end

"""
train_UDE!(simulation::FunctionalInversion; save_every_iter::Bool=false, logger::Union{<: TBLogger, Nothing}=nothing)

Trains UDE based on the current FunctionalInversion.
"""
function train_UDE!(
    simulation::FunctionalInversion;
    save_every_iter::Bool = false,
    logger::Union{<: TBLogger, Nothing} = nothing
    )
    optimizer = simulation.parameters.hyper.optimizer
    iceflow_trained = train_UDE!(simulation, optimizer; save_every_iter=save_every_iter, logger=logger)
    return iceflow_trained
end

"""
BFGS Training
"""
function train_UDE!(
    simulation::FunctionalInversion,
    optimizer::Optim.FirstOrderOptimizer;
    save_every_iter::Bool = false,
    logger::Union{<: TBLogger, Nothing} = nothing
    )

    @info "Training with BFGS optimizer"

    # Create batches for inversion training
    simulation_train_loader = generate_batches(simulation)
    # simulation_batch_ids = train_loader.data[1]

    θ = simulation.model.machine_learning.θ

    # Get the available workers
    # Workers are always the number of allocated cores to Julia minus one
    workers_list = workers()
    if simulation.parameters.simulation.multiprocessing
        @assert length(workers_list) == (simulation.parameters.simulation.workers - 1) "Number of workers does not match"
    end

    # Simplify API for optimization problem and include data loaded in argument for minibatch
    loss_function(_θ, _simulation) = loss_iceflow_transient(_θ, only(_simulation.data), pmap)

    if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
        @assert simulation.parameters.UDE.optim_autoAD == Optimization.AutoZygote() "For the moment only Zygote is supported for the differentiation of the loss function."
    else
        @info "Training with custom $(typeof(simulation.parameters.UDE.grad)) method"
    end
    loss_function_grad!(_dθ, _θ, _simulation) = if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
        grad_loss_iceflow!(_dθ, _θ, only(_simulation), pmap)
    else
        SIA2D_grad!(_dθ, _θ, only(_simulation))
    end
    optf = OptimizationFunction(loss_function, NoAD(), grad=loss_function_grad!)

    optprob = OptimizationProblem(optf, θ, simulation_train_loader)

    # Training diagnosis callback
    cb(θ, l) = let simulation=simulation, logger=logger, save_every_iter=save_every_iter
        callback_diagnosis(θ, l, simulation; save = save_every_iter, tbLogger = logger)
    end

    iceflow_trained = solve(
        optprob,
        simulation.parameters.hyper.optimizer,
        maxiters = simulation.parameters.hyper.epochs,
        allow_f_increases = true,
        callback = cb,
        progress = false
        )

    θ_trained = iceflow_trained.u
    simulation.results.simulation = create_results(θ_trained, simulation, pmap)

    return iceflow_trained
end

"""
ADAM Training
"""
function train_UDE!(
    simulation::FunctionalInversion,
    optimizer::AR;
    save_every_iter::Bool = false,
    logger::Union{<: TBLogger, Nothing} = nothing
    ) where {AR <: Optimisers.AbstractRule}

    @info "Training with ADAM optimizer"

    # Create batches for inversion training
    simulation_train_loader = generate_batches(simulation)

    # The variable θ includes all variables to being optimized, including initial conditions
    θ = simulation.model.machine_learning.θ

    # Get the available workers
    # Workers are always the number of allocated cores to Julia minus one
    workers_list = workers()
    if simulation.parameters.simulation.multiprocessing
        @assert length(workers_list) == (simulation.parameters.simulation.workers - 1) "Number of workers does not match"
    end

    # Simplify API for optimization problem and include data loaded in argument for minibatch
    # glacier_data_batch is a pair of the data sampled (e.g, glacier_data_batch = (id, glacier))
    # _glacier_data_batch has a simulation!
    loss_function(_θ, simulation_loader) = loss_iceflow_transient(_θ, simulation_loader[1], pmap)

    if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
        @assert simulation.parameters.UDE.optim_autoAD == Optimization.AutoZygote() "For the moment only Zygote is supported for the differentiation of the loss function."
    else
        @info "Training with custom $(typeof(simulation.parameters.UDE.grad)) method"
    end
    loss_function_grad!(_dθ, _θ, simulation_loader) = if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
        grad_loss_iceflow!(_dθ, _θ, simulation_loader[1], pmap)
    else
        SIA2D_grad!(_dθ, _θ, simulation_loader[1])
    end
    optf = OptimizationFunction(loss_function, NoAD(), grad=loss_function_grad!)

    optprob = OptimizationProblem(optf, θ, simulation_train_loader)

    # Training diagnosis callback
    cb(θ, l) = let simulation=simulation, logger=logger, save_every_iter=save_every_iter
        callback_diagnosis(θ, l, simulation; save = save_every_iter, tbLogger = logger)
    end

    iceflow_trained = solve(
        optprob,
        simulation.parameters.hyper.optimizer,
        maxiters = simulation.parameters.hyper.epochs,
        callback = cb,
        progress = false
        )

    θ_trained = iceflow_trained.u
    simulation.results.simulation = create_results(θ_trained, simulation, pmap)

    return iceflow_trained
end

"""
    create_results(θ, simulation::FunctionalInversion, mappingFct)

Given the parameters θ, solve the iceflow problem for all the glaciers and aggregate
the results for all of them.
This function is typically used at the end of a training once θ has been optimized
and one wants to run one last forward simulation in order to retrieve statistics
about each of the iceflow problems.

Arguments:
- `θ`: Parameters to use for the forward simulation.
- `simulation::FunctionalInversion`: Simulation structure that contains all the required information about the functional inversion.
- `mappingFct`: Function to use to process the glaciers. Either `map` for a sequential processing or `pmap` for multiprocessing.
"""
function create_results(θ, simulation::FunctionalInversion, mappingFct)
    simulations = generate_simulation_batches(simulation)
    results = mappingFct(simulations) do simulation
        container = FunctionalInversionBinder(simulation, θ)
        [_batch_iceflow_UDE(
            container, glacier_idx,
            define_iceflow_prob(θ, simulation, glacier_idx)
        ) for glacier_idx in 1:length(container.simulation.glaciers)]
    end
    results = merge_batches(results)
    return results
end

"""
    loss_iceflow_transient(θ, simulation::FunctionalInversion, mappingFct)

Given the parameters θ, this function:
1) Solves the iceflow problem for all the glaciers.
2) Computes the loss function defined as the sum of the loss functions for each of the glaciers.
    The loss function of each glacier depends on the type of loss. Refer to `empirical_loss_function` in
    the UDE parameters for more information. The loss function is transient meaning that the state of the
    glacier is compared to a reference at different time steps over the simulated period.
3) Return the value of the loss function.

Arguments:
- `θ`: Parameters to use for the forward simulation.
- `simulation::FunctionalInversion`: Simulation structure that contains all the required information about the functional inversion.
- `mappingFct`: Function to use to process the glaciers. Either `map` for a sequential processing or `pmap` for multiprocessing.
"""
function loss_iceflow_transient(θ, simulation::FunctionalInversion, mappingFct)
    simulations = generate_simulation_batches(simulation)
    losses = mappingFct(
        simulation -> parallel_loss_iceflow_transient(
            θ, simulation,
        ), simulations)
    losses = merge_batches(losses)

    l_H = sum(losses)
    return l_H
end

"""
    grad_loss_iceflow!(dθ, θ, simulation::FunctionalInversion, mappingFct)

Compute the gradient with respect to θ for all the glaciers and assign the result in-place to `dθ`.

Arguments:
- `dθ`: Gradient of the parameters where the computed gradient should be stored.
- `θ`: Parameters to differentiate.
- `simulation::FunctionalInversion`: Simulation structure that contains all the required information about the functional inversion.
- `mappingFct`: Function to use to process the glaciers. Either `map` for a sequential processing or `pmap` for multiprocessing.
"""
function grad_loss_iceflow!(dθ, θ, simulation::FunctionalInversion, mappingFct)
    dθ .= grad_loss_iceflow!(θ, simulation::FunctionalInversion, mappingFct)
end

"""
    grad_loss_iceflow!(θ, simulation::FunctionalInversion, mappingFct)

Compute the gradient with respect to θ for all the glaciers and return the result out-of-place.
See the in-place implementation for more information.
"""
function grad_loss_iceflow!(θ, simulation::FunctionalInversion, mappingFct)
    if simulation.parameters.simulation.use_MB
        @assert simulation.parameters.UDE.optim_autoAD isa NoAD "Differentiation of callbacks with SciMLStruct is not supported by SciMLSensitivity yet. You get this error because you are using MB + gradient computation with SciMLSensitivity."
    end

    simulations = generate_simulation_batches(simulation)
    grads = mappingFct(simulations) do simulation
        [grad_parallel_loss_iceflow!(θ, simulation, glacier_idx) for glacier_idx in 1:length(simulation.glaciers)]
    end
    return sum(merge_batches(grads))
end

"""
    grad_parallel_loss_iceflow!(θ, simulation::FunctionalInversion, glacier_idx::Integer)

Compute the gradient with respect to θ for a particular glacier and return the computed gradient.
This function defines the iceflow problem and then calls Zygote to differentiate `batch_loss_iceflow_transient` with respect to θ.
It uses the SciMLSensitivity implementation under the hood to compute the adjoint of the ODE.
"""
function grad_parallel_loss_iceflow!(θ, simulation::FunctionalInversion, glacier_idx::Integer)
    iceflow_prob = define_iceflow_prob(θ, simulation, glacier_idx)
    ret, = Zygote.gradient(
        _θ -> batch_loss_iceflow_transient(
            FunctionalInversionBinder(simulation, _θ),
            glacier_idx,
            iceflow_prob,
        )[1], θ)
    return ret
end

"""
    parallel_loss_iceflow_transient(θ, simulation::FunctionalInversion)

Loop over a list of glaciers to process.
When multiprocessing is enabled, each call of this function has a dedicated process.
This function calls `batch_loss_iceflow_transient` which returns both the loss and the result structure. The function keeps only the loss.
"""
function parallel_loss_iceflow_transient(θ, simulation::FunctionalInversion)
    return [
        batch_loss_iceflow_transient(
            FunctionalInversionBinder(simulation, θ),
            glacier_idx,
            define_iceflow_prob(θ, simulation, glacier_idx)
            )[1]
        for glacier_idx in 1:length(simulation.glaciers)
        ]
end

"""
    batch_loss_iceflow_transient(
        container::FunctionalInversionBinder,
        glacier_idx::Integer,
        iceflow_prob::ODEProblem,
    )

Solve the ODE, retrieve the results and compute the loss.

Arguments:
- `container::FunctionalInversionBinder`: SciMLStruct that contains the simulation structure and the vector of parameters to optimize.
- `glacier_idx::Integer`: Index of the glacier.
- `iceflow_prob::ODEProblem`: Iceflow problem defined as an ODE with respect to time.
"""
function batch_loss_iceflow_transient(
    container::FunctionalInversionBinder,
    glacier_idx::Integer,
    iceflow_prob::ODEProblem,
)
    result = _batch_iceflow_UDE(container, glacier_idx, iceflow_prob)

    loss_function = container.simulation.parameters.UDE.empirical_loss_function

    glacier = container.simulation.glaciers[glacier_idx]
    H_ref = glacier.thicknessData.H
    t = result.t
    Δt = diff(t)
    H = result.H
    @assert size(H_ref[begin]) == size(H[begin]) "Initial state of reference and predicted ice thickness do not match."
    @assert length(H_ref) == length(H) "Size of reference and prediction datasets do not match."
    @assert t == glacier.thicknessData.t "Reference and prediction need to be evaluated with the same timestamps."

    if loss_uses_ref_velocity(loss_function)
        @assert !isnothing(glacier.velocityData) "Using $(typeof(loss_function)) but no velocityData in the glacier $(glacier.rgi_id)"
        @assert length(glacier.velocityData.date) > 0 "Using $(typeof(loss_function)) but no reference velocity in the results"
    end

    β = 2.0
    l_H = map(2:length(H)) do τ
        normalization = 1.0
        # normalization = std(H_ref[τ][H_ref[τ] .> 0.0])^β
        Hr = @ignore_derivatives(H_ref[τ]) # Ignore this part of the computational graph, otherwise AD fails
        mean_error = loss(
                loss_function,
                H[τ],
                Hr,
                t[τ],
                glacier,
                container.θ,
                container.simulation,
                prod(size(H_ref[τ]))*normalization,
            )
        Δt[τ-1] * mean_error
    end
    return sum(l_H), result
end

"""
    _batch_iceflow_UDE(
        container::FunctionalInversionBinder,
        glacier_idx::Integer,
        iceflow_prob::ODEProblem,
    )

Define the callbacks to be called by the ODE solver, solve the ODE and create the results.
"""
function _batch_iceflow_UDE(
    container::FunctionalInversionBinder,
    glacier_idx::Integer,
    iceflow_prob::ODEProblem,
)
    params = container.simulation.parameters
    glacier = container.simulation.glaciers[glacier_idx]
    step = params.solver.step

    container.simulation.cache = init_cache(container.simulation.model, container.simulation, glacier_idx, container.θ)
    container.simulation.model.machine_learning.θ = container.θ

    # Create mass balance callback
    tstops = Huginn.define_callback_steps(params.simulation.tspan, step)
    params.solver.tstops = tstops

    cb_MB = if params.simulation.use_MB
        # For the moment there is a bug when we use callbacks with SciMLSensitivity for the gradient computation
        mb_action! = let model = container.simulation.model, cache = container.simulation.cache, glacier = glacier, step = step
            function (integrator)
                # Compute mass balance
                glacier.S .= glacier.B .+ integrator.u
                MB_timestep!(cache, model, glacier, step, integrator.t)
                apply_MB_mask!(integrator.u, cache.iceflow)
            end
        end
        # A simulation period is sliced in time windows that are separated by `step`
        # The mass balance is applied at the end of each of the windows
        PeriodicCallback(mb_action!, step; initial_affect=false)
    else
        CallbackSet()
    end

    # Create iceflow law callback
    cb_iceflow = Huginn.build_callback(
        container.simulation.model.iceflow,
        container.simulation.cache.iceflow,
        container.simulation.cache.iceflow.glacier_idx,
        container.θ,
        params.simulation.tspan,
    )

    cb = CallbackSet(cb_MB, cb_iceflow)

    # Run iceflow UDE for this glacier
    iceflow_sol = simulate_iceflow_UDE!(container, cb, iceflow_prob)

    # Compute simulation results
    return Sleipnir.create_results(
        container.simulation, glacier_idx, iceflow_sol, nothing;
        light = !container.simulation.parameters.solver.save_everystep,
    )
end


"""
    simulate_iceflow_UDE!(
        container::FunctionalInversionBinder,
        cb::SciMLBase.DECallback,
        iceflow_prob::ODEProblem,
    )

Make a forward simulation of the iceflow UDE.
"""
function simulate_iceflow_UDE!(
    container::FunctionalInversionBinder,
    cb::SciMLBase.DECallback,
    iceflow_prob::ODEProblem,
)
    params = container.simulation.parameters
    iceflow_prob_remake = remake(iceflow_prob; p = container)
    iceflow_sol = solve(
        iceflow_prob_remake,
        params.solver.solver,
        callback = cb,
        sensealg = params.UDE.sensealg,
        reltol = params.solver.reltol,
        progress = false,
        maxiters = params.solver.maxiters,
        tstops = params.solver.tstops,
    )
    @assert iceflow_sol.retcode==ReturnCode.Success "There was an error in the iceflow solver. Returned code is \"$(iceflow_sol.retcode)\""

    return iceflow_sol
end

"""
    define_iceflow_prob(
        simulation::FunctionalInversion,
        glacier_idx::Integer,
    )

Given a `simulation` struct and a `glacier_idx`, build the iceflow problem that has to be solved in the ODE solver.
In practice, the returned iceflow problem is used inside `simulate_iceflow_UDE!` through `remake`.
The definition of the iceflow problem has to be done outside of the gradient computation, otherwise Zygote fails at differentiating it.
"""
function define_iceflow_prob(
    θ,
    simulation::FunctionalInversion,
    glacier_idx::Integer,
)
    params = simulation.parameters
    # Define initial condition for the inverse simulation
    if haskey(θ, :IC)
        H₀ = evaluate_H₀(
            θ,
            simulation.glaciers[glacier_idx],
            simulation.parameters.UDE.initial_condition_filter
            )
        @assert size(H₀) == size(simulation.glaciers[glacier_idx].H₀)
    else
        H₀ = simulation.glaciers[glacier_idx].H₀
    end
    iceflow_prob = ODEProblem(
        SIA2D_UDE!,
        H₀,
        params.simulation.tspan,
        simulation;
        tstops = params.solver.tstops,
    )
    return iceflow_prob
end

"""
    precompute_all_VJPs_laws!(
        SIA2D_model::SIA2Dmodel,
        SIA2D_cache::SIA2DCache,
        simulation::FunctionalInversion,
        glacier_idx::Integer,
        t::Real,
        θ,
    )

Precomputes the vector-Jacobian products (VJPs) for all laws used in
the SIA2D ice flow model for a given glacier, time, and model parameters.

Depending on which target (`U`, `Y`, or neither) is provided in
`SIA2D_model`, this function checks if the corresponding law supports
VJP precomputation and, if so, triggers the appropriate precompute
routine for that law. If neither `U` nor `Y` is provided, precomputes
VJPs for the `A`, `C`, and `n` laws.

# Arguments
- `SIA2D_model::SIA2Dmodel`: The model containing the configuration and
    laws used for SIA2D ice flow.
- `SIA2D_cache::SIA2DCache`: A cache object holding intermediate values
    and storage relevant for precomputations.
- `simulation::FunctionalInversion`: Simulation object containing global simulation parameters.
- `glacier_idx::Integer`: Index of the glacier being simulated.
- `t::Real`: Current time in the simulation.
- `θ`: Model parameters or state variables for the simulation step.

# Notes
- This routine is intended as a preparatory step for manual adjoint.
- Only laws supporting VJP precomputation are processed.
"""
function precompute_all_VJPs_laws!(
    SIA2D_model::SIA2Dmodel,
    SIA2D_cache::SIA2DCache,
    simulation::FunctionalInversion,
    glacier_idx::Integer,
    t::Real,
    θ,
)
    if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint) || isa(simulation.parameters.UDE.grad, DummyAdjoint)
        return nothing
    end
    if isa(simulation.parameters.UDE.grad.VJP_method, EnzymeVJP)
        return nothing
    end
    if SIA2D_model.U_is_provided
        if is_precomputable_law_VJP(SIA2D_model.U)
            precompute_law_VJP(SIA2D_model.U, SIA2D_cache.U, SIA2D_cache.U_prep_vjps, simulation, glacier_idx, t, θ)
        end
    elseif SIA2D_model.Y_is_provided
        if is_precomputable_law_VJP(SIA2D_model.Y)
            precompute_law_VJP(SIA2D_model.Y, SIA2D_cache.Y, SIA2D_cache.Y_prep_vjps, simulation, glacier_idx, t, θ)
        end
    else
        if is_precomputable_law_VJP(SIA2D_model.A)
            precompute_law_VJP(SIA2D_model.A, SIA2D_cache.A, SIA2D_cache.A_prep_vjps, simulation, glacier_idx, t, θ)
        end
        if is_precomputable_law_VJP(SIA2D_model.C)
            precompute_law_VJP(SIA2D_model.C, SIA2D_cache.C, SIA2D_cache.C_prep_vjps, simulation, glacier_idx, t, θ)
        end
        if is_precomputable_law_VJP(SIA2D_model.n)
            precompute_law_VJP(SIA2D_model.n, SIA2D_cache.n, SIA2D_cache.n_prep_vjps, simulation, glacier_idx, t, θ)
        end
    end
end

"""
Currently just used for Enzyme
"""
function SIA2D_UDE!(_dH::Matrix{<: Real}, _H::Matrix{<: Real}, container::FunctionalInversionBinder, t::Real)
    Huginn.SIA2D!(_dH, _H, container.simulation, t, container.θ)
    return nothing
end
function SIA2D_UDE!(_θ, _dH::Matrix{<: Real}, _H::Matrix{<: Real}, simulation::FunctionalInversion, t::Real)
    Huginn.SIA2D!(_dH, _H, simulation, t, _θ)
    return nothing
end
