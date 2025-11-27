import Huginn.precompute_all_VJPs_laws!, Huginn.run!

"""
    run!(simulation::Inversion)

Run the training process for a given `Inversion` simulation.

# Arguments
- `simulation::Inversion`: The simulation object containing the parameters and settings for the inversion process.

# Description
This function initiates the training of a Universal Differential Equation (UDE) for the provided simulation. It prints a message indicating the start of the training process, calls the `train_UDE!` function to perform the training, and collects the results in `results_list`. The results are intended to be saved using `Sleipnir.save_results_file!`, but this step is currently commented out and will be enabled once the optimization is working. Finally, the garbage collector is triggered to free up memory.

# Notes
- The `Sleipnir.save_results_file!` function call is currently commented out and should be enabled once the optimization process is confirmed to be working.
- The garbage collector is explicitly run using `GC.gc()` to manage memory usage.
"""
function run!(
    simulation::Inversion;
    path::Union{String, Nothing} = nothing,
    file_name::Union{String, Nothing} = nothing,
    save_every_iter::Bool = false,
    path_tb_logger::Union{String, Nothing} = joinpath(
        ODINN.root_dir,
        ".log/",
        Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"),
    ),
)
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
                simulation.model.trainable_components.θ = θ_trained
            end
            sol = train_UDE!(simulation; save_every_iter=save_every_iter, logger=logger)
            # Clear results of previous simulation for fresh start
            if i < length(epochs)
                simulation.results.simulation = Sleipnir.Results{Float64, Int64}[]
            end
        end
    end

    # Setup final results
    simulation.model.trainable_components.θ = sol.u

    simulation.results.stats.niter = length(simulation.results.stats.losses)
    # Final parameters of the optimization
    simulation.results.stats.θ = sol.u
    # Final initial conditions for the simulation
    if haskey(sol.u, :IC)
        simulation.results.stats.initial_conditions = Dict()
        for glacier_id in 1:length(simulation.glaciers)
            glacier = simulation.glaciers[glacier_id]
            simulation.results.stats.initial_conditions[String(glacier.rgi_id)] = evaluate_H₀(
                simulation.model.trainable_components.θ,
                glacier,
                simulation.parameters.UDE.initial_condition_filter,
                glacier_id,
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
    train_UDE!(
        simulation::Inversion;
        save_every_iter::Bool = false,
        logger::Union{<: TBLogger, Nothing} = nothing
    )

Trains UDE based on the current Inversion.
"""
function train_UDE!(
    simulation::Inversion;
    save_every_iter::Bool = false,
    logger::Union{<: TBLogger, Nothing} = nothing
)
    optimizer = simulation.parameters.hyper.optimizer
    iceflow_trained = train_UDE!(simulation, optimizer; save_every_iter=save_every_iter, logger=logger)
    return iceflow_trained
end

"""
BFGS optim
"""
function train_UDE!(
    simulation::Inversion,
    optimizer::Optim.FirstOrderOptimizer;
    save_every_iter::Bool = false,
    logger::Union{<: TBLogger, Nothing} = nothing
    )

    @info "Optimizing with BFGS"

    # Create batches for inversion training
    simulation_train_loader = generate_batches(simulation)
    # simulation_batch_ids = train_loader.data[1]

    θ = simulation.model.trainable_components.θ
    allowed_keys = (:A, :C, :n, :Y, :U)
    @assert length(intersect(keys(θ), allowed_keys))==1 "The vector of parameters θ should contain at most only one of the following keys: $(allowed_keys)"

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
        @info "Optimizing with custom $(typeof(simulation.parameters.UDE.grad)) method"
    end
    loss_function_grad!(_dθ, _θ, _simulation) = if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
        grad_loss_iceflow!(_dθ, _θ, only(_simulation), pmap)
    else
        SIA2D_grad!(_dθ, _θ, only(_simulation))
    end
    optf = OptimizationFunction(loss_function, NoAD(), grad=loss_function_grad!)

    optprob = OptimizationProblem(optf, θ, simulation_train_loader)

    # Optim diagnosis callback
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
ADAM optim
"""
function train_UDE!(
    simulation::Inversion,
    optimizer::AR;
    save_every_iter::Bool = false,
    logger::Union{<: TBLogger, Nothing} = nothing
    ) where {AR <: Optimisers.AbstractRule}

    @info "Optimizing with ADAM"

    # Create batches for inversion training
    simulation_train_loader = generate_batches(simulation)

    # The variable θ includes all variables to being optimized, including initial conditions
    θ = simulation.model.trainable_components.θ
    allowed_keys = (:A, :C, :n, :Y, :U)
    @assert length(intersect(keys(θ), allowed_keys))==1 "The vector of parameters θ should contain at most only one of the following keys: $(allowed_keys)"

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
        @info "Optimizing with custom $(typeof(simulation.parameters.UDE.grad)) method"
    end
    loss_function_grad!(_dθ, _θ, simulation_loader) = if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
        grad_loss_iceflow!(_dθ, _θ, simulation_loader[1], pmap)
    else
        SIA2D_grad!(_dθ, _θ, simulation_loader[1])
    end
    optf = OptimizationFunction(loss_function, NoAD(), grad=loss_function_grad!)

    optprob = OptimizationProblem(optf, θ, simulation_train_loader)

    # Optim diagnosis callback
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
    create_results(θ, simulation::Inversion, mappingFct)

Given the parameters θ, solve the iceflow problem for all the glaciers and aggregate
the results for all of them.
This function is typically used at the end of a training once θ has been optimized
and one wants to run one last forward simulation in order to retrieve statistics
about each of the iceflow problems.

Arguments:
- `θ`: Parameters to use for the forward simulation.
- `simulation::Inversion`: Simulation structure that contains all the required information about the inversion.
- `mappingFct`: Function to use to process the glaciers. Either `map` for a sequential processing or `pmap` for multiprocessing.
"""
function create_results(θ, simulation::Inversion, mappingFct)
    simulation.model.trainable_components.θ = θ
    simulations = generate_simulation_batches(simulation)
    results = mappingFct(simulations) do simulation
        container = InversionBinder(simulation, simulation.model.trainable_components.θ)
        [_batch_iceflow_UDE(
            container, glacier_idx,
            define_iceflow_prob(simulation.model.trainable_components.θ, simulation, glacier_idx)
        ) for glacier_idx in 1:length(container.simulation.glaciers)]
    end
    results = merge_batches(results)
    return results
end

"""
    loss_iceflow_transient(θ, simulation::Inversion, mappingFct)

Given the parameters θ, this function:
1) Solves the iceflow problem for all the glaciers.
2) Computes the loss function defined as the sum of the loss functions for each of the glaciers.
    The loss function of each glacier depends on the type of loss. Refer to `empirical_loss_function` in
    the UDE parameters for more information. The loss function is transient meaning that the state of the
    glacier is compared to a reference at different time steps over the simulated period.
3) Return the value of the loss function.

Arguments:
- `θ`: Parameters to use for the forward simulation.
- `simulation::Inversion`: Simulation structure that contains all the required information about the inversion.
- `mappingFct`: Function to use to process the glaciers. Either `map` for a sequential processing or `pmap` for multiprocessing.
"""
function loss_iceflow_transient(θ, simulation::Inversion, mappingFct)
    simulation.model.trainable_components.θ = θ
    simulations = generate_simulation_batches(simulation)
    losses = mappingFct(
        simulation -> parallel_loss_iceflow_transient(
            simulation.model.trainable_components.θ, simulation,
        ), simulations)
    losses = merge_batches(losses)
    return sum(losses)
end

"""
    grad_loss_iceflow!(dθ, θ, simulation::Inversion, mappingFct)

Compute the gradient with respect to θ for all the glaciers and assign the result in-place to `dθ`.

Arguments:
- `dθ`: Gradient of the parameters where the computed gradient should be stored.
- `θ`: Parameters to differentiate.
- `simulation::Inversion`: Simulation structure that contains all the required information about the inversion.
- `mappingFct`: Function to use to process the glaciers. Either `map` for a sequential processing or `pmap` for multiprocessing.
"""
function grad_loss_iceflow!(dθ, θ, simulation::Inversion, mappingFct)
    dθ .= grad_loss_iceflow!(θ, simulation::Inversion, mappingFct)
end

"""
    grad_loss_iceflow!(θ, simulation::Inversion, mappingFct)

Compute the gradient with respect to θ for all the glaciers and return the result out-of-place.
See the in-place implementation for more information.
"""
function grad_loss_iceflow!(θ, simulation::Inversion, mappingFct)
    if simulation.parameters.simulation.use_MB
        @assert simulation.parameters.UDE.optim_autoAD isa NoAD "Differentiation of callbacks with SciMLStruct is not supported by SciMLSensitivity yet. You get this error because you are using MB + gradient computation with SciMLSensitivity."
    end

    simulation.model.trainable_components.θ = θ
    simulations = generate_simulation_batches(simulation)
    grads = mappingFct(simulations) do simulation
        [grad_parallel_loss_iceflow!(simulation.model.trainable_components.θ, simulation, glacier_idx) for glacier_idx in 1:length(simulation.glaciers)]
    end
    return sum(merge_batches(grads))
end

"""
    grad_parallel_loss_iceflow!(θ, simulation::Inversion, glacier_idx::Integer)

Compute the gradient with respect to θ for a particular glacier and return the computed gradient.
This function defines the iceflow problem and then calls Zygote to differentiate `batch_loss_iceflow_transient` with respect to θ.
It uses the SciMLSensitivity implementation under the hood to compute the adjoint of the ODE.
"""
function grad_parallel_loss_iceflow!(θ, simulation::Inversion, glacier_idx::Integer)
    iceflow_prob = define_iceflow_prob(θ, simulation, glacier_idx)
    ret, = Zygote.gradient(
        _θ -> batch_loss_iceflow_transient(
            InversionBinder(simulation, _θ),
            glacier_idx,
            iceflow_prob,
        )[1], θ)
    return ret
end

"""
    parallel_loss_iceflow_transient(θ, simulation::Inversion)

Loop over a list of glaciers to process.
When multiprocessing is enabled, each call of this function has a dedicated process.
This function calls `batch_loss_iceflow_transient` which returns both the loss and the result structure. The function keeps only the loss.
"""
function parallel_loss_iceflow_transient(θ, simulation::Inversion)
    return [
        batch_loss_iceflow_transient(
            InversionBinder(simulation, θ),
            glacier_idx,
            define_iceflow_prob(θ, simulation, glacier_idx)
            )[1]
        for glacier_idx in 1:length(simulation.glaciers)
        ]
end

"""
    batch_loss_iceflow_transient(
        container::InversionBinder,
        glacier_idx::Integer,
        iceflow_prob::ODEProblem,
    )

Solve the ODE, retrieve the results and compute the loss.

Arguments:
- `container::InversionBinder`: SciMLStruct that contains the simulation structure and the vector of parameters to optimize.
- `glacier_idx::Integer`: Index of the glacier.
- `iceflow_prob::ODEProblem`: Iceflow problem defined as an ODE with respect to time.
"""
function batch_loss_iceflow_transient(
    container::InversionBinder,
    glacier_idx::Integer,
    iceflow_prob::ODEProblem,
)
    result = _batch_iceflow_UDE(container, glacier_idx, iceflow_prob)

    loss_function = container.simulation.parameters.UDE.empirical_loss_function

    glacier = container.simulation.glaciers[glacier_idx]
    t = result.t
    H = result.H

    if loss_uses_velocity(loss_function)
        @assert !isnothing(glacier.velocityData) "Using $(typeof(loss_function)) but no velocityData in the glacier $(glacier.rgi_id)"
        @assert length(glacier.velocityData.date) > 0 "Using $(typeof(loss_function)) but no reference velocity in the results"
    end

    # Discretization for the ice thickness loss term
    tH_ref = tdata(glacier.thicknessData) # If thicknessData is nothing, then tH_ref is an empty vector
    ΔtH = diff(tH_ref)
    useThickness = length(tH_ref)>0
    H_ref = useThickness ? glacier.thicknessData.H : nothing

    # Discretization for the surface velocity loss term
    tV_ref = tdata(glacier.velocityData, container.simulation.parameters.simulation.mapping) # If velocityData is nothing, then tV_ref is an empty vector
    ΔtV = diff(tV_ref)
    useVelocity = length(tV_ref)>0
    Vabs_ref = useVelocity ? glacier.velocityData.vabs : nothing
    Vx_ref = useVelocity ? glacier.velocityData.vx : nothing
    Vy_ref = useVelocity ? glacier.velocityData.vy : nothing

    # Discretization provided to the loss as a named tuple with the discretization for each term
    Δt_HV = (; H=ΔtH, V=ΔtV)

    if useThickness
        @assert size(H[begin]) == size(H_ref[begin]) "Size of reference and prediction datasets do not match."
    end
    if useVelocity
        @assert size(H[begin]) == size(Vabs_ref[begin]) "Size of reference and prediction datasets do not match."
    end

    β = 2.0
    losses = map(1:length(H)) do τ
        normalization = 1.0
        # normalization = std(H_ref[τ][H_ref[τ] .> 0.0])^β

        tj = t[τ]
        indThickness = findfirst(==(tj), tH_ref)
        indVelocity = findfirst(==(tj), tV_ref)

        # Ignore these parts of the computational graph, otherwise AD fails
        Hr = @ignore_derivatives(isnothing(indThickness) ? nothing : H_ref[indThickness])
        Vr = @ignore_derivatives(isnothing(indVelocity) ? nothing : Vabs_ref[indVelocity])
        Vxr = @ignore_derivatives(isnothing(indVelocity) ? nothing : Vx_ref[indVelocity])
        Vyr = @ignore_derivatives(isnothing(indVelocity) ? nothing : Vy_ref[indVelocity])
        Δtj = @ignore_derivatives((;
            H=isnothing(indThickness) ? 0.0 : safe_slice(Δt_HV.H, indThickness-1),
            V=isnothing(indVelocity) ? 0.0 : safe_slice(Δt_HV.V, indVelocity-1),
        ))

        loss(
            loss_function,
            H[τ],
            Hr,
            Vr, Vxr, Vyr,
            t[τ],
            glacier_idx,
            container.θ,
            container.simulation,
            prod(size(H[τ]))*normalization,
            Δtj,
        )
    end
    return sum(losses), result
end

"""
    _batch_iceflow_UDE(
        container::InversionBinder,
        glacier_idx::Integer,
        iceflow_prob::ODEProblem,
    )

Define the callbacks to be called by the ODE solver, solve the ODE and create the results.
"""
function _batch_iceflow_UDE(
    container::InversionBinder,
    glacier_idx::Integer,
    iceflow_prob::ODEProblem,
)
    params = container.simulation.parameters
    glacier = container.simulation.glaciers[glacier_idx]
    step = params.solver.step
    step_MB = params.simulation.step_MB

    container.simulation.cache = init_cache(container.simulation.model, container.simulation, glacier_idx, container.θ)
    container.simulation.model.trainable_components.θ = container.θ

    # Define tstops
    tstops = Huginn.define_callback_steps(params.simulation.tspan, step)
    tstops = unique(vcat(tstops, params.solver.tstops)) # Merge time steps controlled by `step` with the user provided time steps
    tstopsIceThickness = tdata(glacier.thicknessData)
    tstopsVelocity = tdata(glacier.velocityData, params.simulation.mapping)
    tstopsDiscreteLoss = discreteLossSteps(params.UDE.empirical_loss_function, params.simulation.tspan)
    tstops = sort(unique(vcat(tstops, tstopsIceThickness, tstopsVelocity, tstopsDiscreteLoss)))

    # Create mass balance callback
    cb_MB = if params.simulation.use_MB
        # For the moment there is a bug when we use callbacks with SciMLSensitivity for the gradient computation
        mb_action! = let model = container.simulation.model, cache = container.simulation.cache, glacier = glacier, step_MB = step_MB
            function (integrator)
                # Compute mass balance
                glacier.S .= glacier.B .+ integrator.u
                MB_timestep!(cache, model, glacier, step_MB, integrator.t)
                apply_MB_mask!(integrator.u, cache.iceflow)
            end
        end
        # A simulation period is sliced in time windows that are separated by `step_MB`
        # The mass balance is applied at the end of each of the windows
        PeriodicCallback(mb_action!, step_MB; initial_affect=false)
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
    iceflow_sol = simulate_iceflow_UDE!(container, cb, iceflow_prob, tstops)

    # Compute simulation results
    return Sleipnir.create_results(
        container.simulation, glacier_idx, iceflow_sol, tstops,
    )
end


"""
    simulate_iceflow_UDE!(
        container::InversionBinder,
        cb::SciMLBase.DECallback,
        iceflow_prob::ODEProblem,
        tstops,
    )

Make a forward simulation of the iceflow UDE.
"""
function simulate_iceflow_UDE!(
    container::InversionBinder,
    cb::SciMLBase.DECallback,
    iceflow_prob::ODEProblem,
    tstops,
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
        tstops = tstops,
    )
    @assert iceflow_sol.retcode==ReturnCode.Success "There was an error in the iceflow solver. Returned code is \"$(iceflow_sol.retcode)\""

    return iceflow_sol
end

"""
    define_iceflow_prob(
        simulation::Inversion,
        glacier_idx::Integer,
    )

Given a `simulation` struct and a `glacier_idx`, build the iceflow problem that has to be solved in the ODE solver.
In practice, the returned iceflow problem is used inside `simulate_iceflow_UDE!` through `remake`.
The definition of the iceflow problem has to be done outside of the gradient computation, otherwise Zygote fails at differentiating it.
"""
function define_iceflow_prob(
    θ,
    simulation::Inversion,
    glacier_idx::Integer,
)
    params = simulation.parameters
    # Define initial condition for the inverse simulation
    if haskey(θ, :IC)
        H₀ = evaluate_H₀(
            θ,
            simulation.glaciers[glacier_idx],
            simulation.parameters.UDE.initial_condition_filter,
            glacier_idx,
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
        simulation::Inversion,
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
- `simulation::Inversion`: Simulation object containing global simulation parameters.
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
    simulation::Inversion,
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
function SIA2D_UDE!(_dH::Matrix{<: Real}, _H::Matrix{<: Real}, container::InversionBinder, t::Real)
    Huginn.SIA2D!(_dH, _H, container.simulation, t, container.θ)
    return nothing
end
function SIA2D_UDE!(_θ, _dH::Matrix{<: Real}, _H::Matrix{<: Real}, simulation::Inversion, t::Real)
    Huginn.SIA2D!(_dH, _H, simulation, t, _θ)
    return nothing
end
