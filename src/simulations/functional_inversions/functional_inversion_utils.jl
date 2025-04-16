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
function run!(simulation::FunctionalInversion)

    println("Running training of UDE...\n")

    # Set expected total number of epochs from beginning for the callback
    simulation.stats.niter = sum(simulation.parameters.hyper.epochs)

    if !(typeof(simulation.parameters.hyper.optimizer) <: Vector)
        # One single optimizer
        sol = train_UDE!(simulation)
    else
        # Multiple optimizers
        optimizers = simulation.parameters.hyper.optimizer
        epochs = simulation.parameters.hyper.epochs
        @assert length(optimizers) == length(optimizers) "Provide number of epochs as a vector with the same length of optimizers"
        for i in 1:length(epochs)
            # Construct a new simulation for each optimizer
            simulation.parameters.hyper.optimizer = optimizers[i]
            simulation.parameters.hyper.epochs = epochs[i] - 1
            if i !== 1
                θ_trained = sol.u
                simulation.model.machine_learning.θ = θ_trained
            end
            sol = train_UDE!(simulation)
        end
    end

    # Setup final results
    simulation.stats.niter = length(simulation.stats.losses)
    # simulation.stats.retcode = sol.
    simulation.stats.θ = sol.u

    # TODO: Save when optimization is working
    # Sleipnir.save_results_file!(results_list, simulation)

    @everywhere GC.gc() # run garbage collector

end

"""
train_UDE!(simulation::FunctionalInversion)

Trains UDE based on the current FunctionalInversion.
"""
function train_UDE!(simulation::FunctionalInversion)
    optimizer = simulation.parameters.hyper.optimizer
    # for i in 1:length(simulation.glaciers)
    #     if !isnothing(simulation.model.iceflow[])
    #     @assert size(simulation.model.iceflow[i].S) == (simulation.glaciers[i].nx, simulation.glaciers[i].ny) "Glacier and model simulation are non sync: $(size(simulation.model.iceflow[i].S)) != ($(simulation.glaciers[i].nx), $(simulation.glaciers[i].ny))" 
    # end
    iceflow_trained = train_UDE!(simulation, optimizer)
    return iceflow_trained
end

"""
BFGS Training
"""
function train_UDE!(simulation::FunctionalInversion, optimizer::Optim.FirstOrderOptimizer)

    @info "Trainign with BFGS optimizer"

    # Create batches for inversion training
    # @infiltrate
    simulation_train_loader = generate_batches(simulation)
    # simulation_batch_ids = train_loader.data[1]

    θ = simulation.model.machine_learning.θ

    # Get the available workers
    # Workers are always the number of allocated cores to Julia minus one
    workers_list = workers()
    if simulation.parameters.simulation.multiprocessing
        @assert length(workers_list) == (simulation.parameters.simulation.workers-1) "Number of workers does not match"
    end

    # Simplify API for optimization problem and include data loaded in argument for minibatch
    loss_function(_θ, _simulation) = loss_iceflow_transient(_θ, only(_simulation.data))

    if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
        # Enzyme.API.strictAliasing!(false)
        optf = OptimizationFunction(loss_function, simulation.parameters.UDE.optim_autoAD)
    else
        @info "Training with custom $(typeof(simulation.parameters.UDE.grad)) method"

        loss_function_grad!(_dθ, _θ, _simulation) = SIA2D_grad!(_dθ, _θ, only(_simulation))

        optf = OptimizationFunction(loss_function, NoAD(), grad=loss_function_grad!)
    end

    # optprob = OptimizationProblem(optf, θ, (simultion_batch_ids))
    optprob = OptimizationProblem(optf, θ, simulation_train_loader)

    # Plot callback
    cb_plots = (θ, l) -> false
    # TODO: Do we need this following code?
    # if isnothing(simulation.parameters.UDE.target)
    #     cb_plots = (θ, l) -> false
    # elseif simulation.parameters.UDE.target == :A
    #     cb_plots = (θ, l) -> false
    #     # This other option returns weird error right now, commenting for now
    #     # cb_plots(θ, l) = callback_plots_A(θ, l, simulation) # TODO: make this more customizable
    # else
    #     @error "Simulation target not defined."
    # end
    # Training diagnosis callback
    cb_diagnosis(θ, l) = callback_diagnosis(θ, l, only(simulation_train_loader.data))

    # Combined callback
    cb(θ, l) = CallbackOptimizationSet(θ, l; callbacks=(cb_plots, cb_diagnosis))

    println("Training iceflow UDE...")

    iceflow_trained = solve(optprob,
                            simulation.parameters.hyper.optimizer,
                            maxiters=simulation.parameters.hyper.epochs,
                            allow_f_increases=true,
                            callback=cb,
                            progress=false)

    return iceflow_trained
end

"""
ADAM Training
"""
function train_UDE!(simulation::FunctionalInversion, optimizer::AR) where {AR <: Optimisers.AbstractRule}

    @info "Training with ADAM optimizer"
    # Create batches for inversion training
    simulation_train_loader = generate_batches(simulation)

    θ = simulation.model.machine_learning.θ

    # Get the available workers
    # Workers are always the number of allocated cores to Julia minus one
    workers_list = workers()
    if simulation.parameters.simulation.multiprocessing
        @assert length(workers_list) == (simulation.parameters.simulation.workers-1) "Number of workers does not match"
    end

    # Simplify API for optimization problem and include data loaded in argument for minibatch
    # glacier_data_batch is a pair of the data sampled (e.g, glacier_data_batch = (id, glacier))
    # _glacier_data_batch has a simulation!
    loss_function(_θ, simulation_loader) = loss_iceflow_transient(_θ, simulation_loader[1])

    if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint)
        # Enzyme.API.strictAliasing!(false)
        optf = OptimizationFunction(loss_function, simulation.parameters.UDE.optim_autoAD)
    else
        @info "Training with custom $(typeof(simulation.parameters.UDE.grad)) method"

        loss_function_grad!(_dθ, _θ, simulation_loader) = SIA2D_grad!(_dθ, _θ, simulation_loader[1])

        optf = OptimizationFunction(loss_function, NoAD(), grad=loss_function_grad!)
    end

    optprob = OptimizationProblem(optf, θ, simulation_train_loader)

    # Plot callback
    cb_plots = (θ, l) -> false
    # TODO: Do we need this following code?
    # if isnothing(simulation.parameters.UDE.target)
    #     cb_plots = (θ, l) -> false
    # elseif simulation.parameters.UDE.target == :A
    #     cb_plots = (θ, l) -> false
    #     # This other option returns weird error right now, commenting for now
    #     # cb_plots(θ, l) = callback_plots_A(θ, l, simulation) # TODO: make this more customizable
    # else
    #     @error "Simulation target not defined."
    # end
    # Training diagnosis callback
    cb_diagnosis(θ, l) = callback_diagnosis(θ, l, simulation)
    # Combined callback
    cb(θ, l) = CallbackOptimizationSet(θ, l; callbacks=(cb_plots, cb_diagnosis))

    println("Training iceflow UDE...")

    iceflow_trained = solve(optprob,
                            simulation.parameters.hyper.optimizer,
                            maxiters=simulation.parameters.hyper.epochs,
                            callback=cb,
                            progress=false)

    return iceflow_trained
end

function loss_iceflow_transient(θ, simulation::FunctionalInversion)

    predict_iceflow!(θ, simulation)

    loss_function = simulation.parameters.UDE.empirical_loss_function

    l_H = 0.0

    for i in 1:length(simulation.glaciers)

        # Reference cames from data
        # Complete this with feeded data on glacier object
        H_ref = only(simulation.glaciers[i].data).H
        # Prediction comes from simulation
        # batch_results_id = Sleipnir.get_result_id_from_rgi(batch_id, simulation)
        t = simulation.results[i].t
        Δt = diff(t)
        H = simulation.results[i].H
        @assert length(H_ref) == length(H) "Reference and Prediction datasets need to be evaluated in same timestamps."
        @assert size(H_ref[begin]) == size(H[begin])

        β = 2.0
        for τ in 2:length(H)
            normalization = 1.0
            # normalization = std(H_ref[τ][H_ref[τ] .> 0.0])^β
            mean_error = loss(loss_function, H[τ], H_ref[τ]; normalization=prod(size(H_ref[τ]))*normalization)
            l_H += Δt[τ-1] * mean_error
        end
    end

    return l_H

end

function loss_iceflow(θ, simulation::FunctionalInversion)

    # simulation.model.machine_learning.θ = θ # update model parameters
    predict_iceflow!(θ, simulation)

    loss = simulation.parameters.UDE.empirical_loss_function

    # Compute loss function for the full batch
    let l_V = 0.0, l_H =  0.0
    for result in simulation.results
        # Get ice thickness from the reference dataset
        H_ref = result.H_glathida
        # isnothing(H_ref) ? continue : nothing
        # Get ice velocities for the reference dataset
        Vx_ref = result.Vx_ref
        Vy_ref = result.Vy_ref
        # Get ice thickness from the UDE predictions
        H = result.H
        # Get ice velocities prediction from the UDE
        Vx_pred = result.Vx
        Vy_pred = result.Vy

        if simulation.parameters.UDE.scale_loss
            # Ice thickness
            if !isnothing(H_ref)
                normHref = mean(H_ref.^2)^0.5
                l_H_loc = loss(H, H_ref)
                l_H += normHref^(-1) * l_H_loc
            end
            # Ice surface velocities
            normVref = mean(Vx_ref.^2 .+ Vy_ref.^2)^0.5
            # l_V_loc = loss(Vx_pred, inn1(Vx_ref)) + loss(Vy_pred, inn1(Vy_ref))
            l_V_loc = loss(Vx_pred, Vx_ref) + loss(Vy_pred, Vy_ref)
            l_V += normVref^(-1) * l_V_loc
        else
            # Ice thickness
            if !isnothing(H_ref)
                l_H += loss(H[H_ref .!= 0.0], H_ref[H_ref.!= 0.0])
            end
            # Ice surface velocities
            l_V += loss(V_pred[V_ref .!= 0.0], V_ref[V_ref.!= 0.0])
        end
    end # for

    loss_type = simulation.parameters.UDE.loss_type
    @assert (loss_type == "H" || loss_type == "V" || loss_type == "HV") "Invalid `loss_type`. Needs to be 'H', 'V' or 'HV'"
    if loss_type == "H"
        l_tot = l_H/length(simulation.results)
    elseif loss_type == "V"
        l_tot = l_V/length(simulation.results)
    elseif loss_type == "HV"
        l_tot = (l_V + l_H)/length(simulation.results)
    end

    return l_tot
    end # let
end

function predict_iceflow!(θ, simulation::FunctionalInversion)
    simulations = ODINN.generate_simulation_batches(simulation)
    results = pmap(simulation -> batch_iceflow_UDE(θ, simulation), simulations)
    simulation.results = ODINN.merge_batches(results)
end


function batch_iceflow_UDE(θ, simulation::FunctionalInversion)
    return [_batch_iceflow_UDE(θ, simulation, batch_id) for batch_id in 1:length(simulation.glaciers)]
end

function _batch_iceflow_UDE(θ, simulation::FunctionalInversion, batch_id::I) where {I <: Integer}

    iceflow_model = simulation.model.iceflow[batch_id]
    params = simulation.parameters
    # batch_id = Sleipnir.Int(batch_id)
    glacier = simulation.glaciers[batch_id]

    if !isnothing(iceflow_model.S)
        @assert size(iceflow_model.S) == (glacier.nx, glacier.ny) "Glacier and model simulation are non sync: $(size(iceflow_model.S)) != ($(glacier.nx), $(glacier.ny))"
    else
        # TODO: change API of inicitalize glacier
        initialize_iceflow_model!(iceflow_model, batch_id, glacier, params)
    end

    tstops = Huginn.define_callback_steps(params.simulation.tspan, params.solver.step)
    params.solver.tstops = tstops
    stop_condition(u,t,integrator) = Sleipnir.stop_condition_tstops(u,t,integrator, tstops)
    function action!(integrator)
        if params.simulation.use_MB
            # Compute mass balance
            @warn "Changes in API not propagated for MB=true"
            @ignore_derivatives begin
                MB_timestep!(simulation.model, glacier, params.solver.step, integrator.t)
                apply_MB_mask!(integrator.u, glacier, simulation.model)
            end
        end
        # Apply parametrization
        # TODO: Do we need this UDE_parametrization here?
        # apply_UDE_parametrization!(θ, simulation, integrator, batch_id)
    end

    cb_MB = DiscreteCallback(stop_condition, action!)

    # Run iceflow UDE for this glacier
    du = params.simulation.use_iceflow ? Huginn.SIA2D! : Huginn.noSIA2D
    iceflow_sol = simulate_iceflow_UDE!(θ, simulation, cb_MB, batch_id; du = du)

    # Update simulation results
    result = Sleipnir.create_results(simulation, batch_id, iceflow_sol, nothing; light=simulation.parameters.simulation.light, batch_id = batch_id)
    return result

end


"""
function simulate_iceflow_UDE!(
    θ,
    simulation::SIM,
    cb::DiscreteCallback,
    batch_id::I;
    du = Huginn.SIA2D) where {I <: Integer, SIM <: Simulation}

Make forward simulation of the iceflow UDE determined in `du`.
"""
function simulate_iceflow_UDE!(
    θ,
    simulation::SIM,
    cb::DiscreteCallback,
    batch_id::I;
    du = Huginn.SIA2D) where {I <: Integer, SIM <: Simulation}

    model = simulation.model
    params = simulation.parameters

    # TODO: make this more general
    # apply_UDE_parametrization!(θ, simulation, nothing, batch_id) # Apply the parametrization otherwise the physical values are wrong between the beginning of the simulation and the first callback
    # TODO: check everywhere else this function is needed
    SIA2D_UDE_closure(H, θ, t) = SIA2D_UDE(H, θ, t, simulation, batch_id)

    iceflow_prob = ODEProblem(SIA2D_UDE_closure, model.iceflow[batch_id].H₀, params.simulation.tspan, θ; tstops=params.solver.tstops)

    iceflow_sol = solve(iceflow_prob,
                        params.solver.solver,
                        callback=cb,
                        sensealg=params.UDE.sensealg,
                        reltol=params.solver.reltol,
                        progress=false)

    # Compute average ice surface velocities for the simulated period
    model.iceflow[batch_id].H = iceflow_sol.u[end]
    model.iceflow[batch_id].H = ifelse.(model.iceflow[batch_id].H .> 0.0, model.iceflow[batch_id].H , 0.0)

    # Average surface velocity
    Huginn.avg_surface_V(simulation; batch_id = batch_id)

    glacier = simulation.glaciers[batch_id]

    # Surface topography
    model.iceflow[batch_id].S = glacier.B .+ model.iceflow[batch_id].H

    return iceflow_sol
end

"""
Wrapper to pass a parametrization to the SIA2D
"""
function SIA2D_UDE(H::Matrix{R}, θ, t::R, simulation::SIM, batch_id::I) where {R <: Real, I <: Integer, SIM <: Simulation}

    if simulation.model.machine_learning.target.name == :A
        if isnothing(batch_id)
            simulation.model.iceflow.D = nothing
        else
            simulation.model.iceflow[batch_id].D = nothing # We pick the right iceflow model for this glacier
        end
        # simulation.model.iceflow = false
        apply_UDE_parametrization!(θ, simulation, nothing, batch_id) # Apply the parametrization otherwise the physical values are wrong between the beginning of the simulation and the first callback
    elseif simulation.model.machine_learning.target.name == :D
        glacier = simulation.glaciers[batch_id]

        ### Compute some stuff we need in order to compute diffusivity
        Δx = glacier.Δx
        Δy = glacier.Δy
        B = glacier.B

        # TODO: Make simple code to put ∇S inside dual grid
        S = B .+ H
        # All grid variables computed in a staggered grid
        # Compute surface gradients on edges
        S = B .+ H
        dSdx = Huginn.diff_x(S) / Δx
        dSdy = Huginn.diff_y(S) / Δy
        ∇Sx = Huginn.avg_y(dSdx)
        ∇Sy = Huginn.avg_x(dSdy)
        # Compute slope
        ∇S = (∇Sx.^2 .+ ∇Sy.^2).^(1/2)
        # Compute average ice thickness
        H̄ = Huginn.avg(H)

        # Compute diffusivity based on learning target in dual grid
        D = simulation.model.machine_learning.target.D(
            H = H̄, ∇S = ∇S, θ = θ,
            ice_model = simulation.model.iceflow[batch_id],
            ml_model = simulation.model.machine_learning,
            glacier = glacier,
            params = simulation.parameters
        )
        # Update value of D in iceflow model used inside SIA
        # diffusivity_provided = true
        # simulation.model.iceflow[1].D = D
        if isnothing(batch_id)
            simulation.model.iceflow.D = D
        else
            simulation.model.iceflow[batch_id].D = D # We pick the right iceflow model for this glacier
        end
    else
        @error "Target UDE parametrization not provided"
    end
    return Huginn.SIA2D(H, simulation, t; batch_id = batch_id) #, diffusivity_provided = diffusivity_provided)
end

"""
Functional inversion functions
"""

function apply_UDE_parametrization_enzyme!(θ, simulation::FunctionalInversion, smodel::StatefulLuxLayer, batch_id::I) where {I <: Integer}
    # We load the ML model with the parameters
    smodel.ps = θ.θ
    smodel.st = simulation.model.machine_learning.st
    # smodel = StatefulLuxLayer{true}(simulation.model.machine_learning.architecture, θ.θ, simulation.model.machine_learning.st)

    # We generate the ML parametrization based on the target
    if simulation.model.machine_learning.target.name == :A
        # @show predict_A̅(smodel, [mean(simulation.glaciers[batch_id].climate.longterm_temps)])
        min_NN = simulation.parameters.physical.minA
        max_NN = simulation.parameters.physical.maxA
        simulation.model.iceflow[batch_id].A .= predict_A̅(smodel, [mean(simulation.glaciers[batch_id].climate.longterm_temps)], (min_NN, max_NN))
    else
        @error "Simulation target not specified"
    end
end

function apply_UDE_parametrization_enzyme(θ, simulation::FunctionalInversion, smodel::StatefulLuxLayer, batch_id::I) where {I <: Integer}
    # We load the ML model with the parameters
    smodel.ps = θ.θ
    smodel.st = simulation.model.machine_learning.st
    # smodel = StatefulLuxLayer{true}(simulation.model.machine_learning.architecture, θ.θ, simulation.model.machine_learning.st)

    # We generate the ML parametrization based on the target
    if simulation.model.machine_learning.target.name == :A
        min_NN = simulation.parameters.physical.minA
        max_NN = simulation.parameters.physical.maxA
        return predict_A̅(smodel, [mean(simulation.glaciers[batch_id].climate.longterm_temps)], (min_NN, max_NN))[1]
    else
        @error "Simulation target not specified"
    end
end

function apply_UDE_parametrization!(θ, simulation::FunctionalInversion, integrator, batch_id::I) where {I <: Integer}
    # We load the ML model with the parameters
    model = simulation.model.machine_learning.architecture
    st = simulation.model.machine_learning.st
    smodel = StatefulLuxLayer{true}(model, θ.θ, st)

    # We generate the ML parametrization based on the target
    if simulation.model.machine_learning.target.name == :A
        T_mean = mean(simulation.glaciers[batch_id].climate.longterm_temps)
        min_NN = simulation.parameters.physical.minA
        max_NN = simulation.parameters.physical.maxA
        A = predict_A̅(smodel, [T_mean], (min_NN, max_NN))[1]
        simulation.model.iceflow[batch_id].A[] = A
        # @info "Value of A used in UDE simulation:"
    elseif simulation.model.machine_learning.target.name == :D
        min_NN = simulation.parameters.physical.minA
        max_NN = simulation.parameters.physical.maxA
        # Predict value of A based on Temp and H
        T_mean = mean(simulation.glaciers[batch_id].climate.longterm_temps)
        A_space = predict_D(smodel, T_mean, H, (min_NN, max_NN))
        return A_space
    else
        @error "Simulation target ='$(simulation.parameters.UDE.target.name)' not specified"
    end
end

function apply_UDE_parametrization(θ, simulation::FunctionalInversion, T::F) where {F <: AbstractFloat}
    # We load the ML model with the parameters
    model = simulation.model.machine_learning.architecture
    st = simulation.model.machine_learning.st
    smodel = StatefulLuxLayer{true}(model, θ.θ, st)

    # We generate the ML parametrization based on the target
    if simulation.model.machine_learning.target.name == :A
        min_NN = simulation.parameters.physical.minA
        max_NN = simulation.parameters.physical.maxA
        A = predict_A̅(smodel, [T], (min_NN, max_NN))[1]
        return A
    elseif simulation.model.machine_learning.target.name == :D
        min_NN = params.physical.minA
        max_NN = params.physical.maxA
        # Predict value of A based on Temp and H
        T_mean = mean(glacier.climate.longterm_temps)
        A_space = predict_D(smodel, T_mean, H, (min_NN, max_NN))
        return A_space
    else
        @error "Simulation target ='$(simulation.parameters.UDE.target.name)' not specified"
    end
end

function apply_UDE_parametrization(θ, simulation::FunctionalInversion, batch_id::I) where {I <: Integer}
    # We load the ML model with the parameters
    model = simulation.model.machine_learning.architecture
    st = simulation.model.machine_learning.st
    smodel = StatefulLuxLayer{true}(model, θ.θ, st)

    # We generate the ML parametrization based on the target
    if simulation.model.machine_learning.target.name == :A
        min_NN = simulation.parameters.physical.minA
        max_NN = simulation.parameters.physical.maxA
        A = predict_A̅(smodel, [mean(simulation.glaciers[batch_id].climate.longterm_temps)], (min_NN, max_NN))[1]
        return A
    else
        @error "Simulation target not specified"
    end
end
