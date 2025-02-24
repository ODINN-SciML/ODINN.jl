

"""
run!(simulation::FunctionalInversion)

In-place run of the model. 
"""
function run!(simulation::FunctionalInversion)

    println("Running training of UDE...\n")
    results_list = train_UDE!(simulation)

    # TODO: Save when optimization is working
    # Sleipnir.save_results_file!(results_list, simulation)

    @everywhere GC.gc() # run garbage collector

end

"""
train_UDE!(simulation::FunctionalInversion) 

Trains UDE based on the current FunctionalInversion.
"""
function train_UDE!(simulation::FunctionalInversion) 
    
    # Create batches for inversion training 
    train_batches = generate_batches(simulation)
    θ = simulation.model.machine_learning.θ

    if isnothing(simulation.parameters.UDE.grad)
        optf = OptimizationFunction((θ, _, batch_ids, rgi_ids)->loss_iceflow(θ, batch_ids, simulation), simulation.parameters.UDE.optim_autoAD)
    else
        print("Using custom grad function.\n")
        grad(f, U) = rand(Float64, dims(U))
        optf = OptimizationFunction((θ, _, batch_ids, rgi_ids)->loss_iceflow(θ, batch_ids, simulation), NoAD(), grad=grad)
    end
    optprob = OptimizationProblem(optf, θ)
    
    if simulation.parameters.UDE.target == "A"
        cb_plots(θ, l) = callback_plots_A(θ, l, simulation) # TODO: make this more customizable 
    end
  
    println("Training iceflow UDE...")
    
    iceflow_trained = solve(optprob, 
                            simulation.parameters.hyper.optimizer, 
                            ncycle(train_batches, simulation.parameters.hyper.epochs), allow_f_increases=true,
                            callback=cb_plots,
                            progress=true)

    return iceflow_trained
end

function loss_iceflow(θ, batch_ids::Vector{I}, simulation::FunctionalInversion) where {I <: Integer} 

    # simulation.model.machine_learning.θ = θ # update model parameters
    predict_iceflow!(θ, simulation, batch_ids)

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
                l_H_loc = Flux.Losses.mse(H, H_ref; agg=mean) 
                l_H += normHref^(-1) * l_H_loc
            end
            # Ice surface velocities
            normVref = mean(Vx_ref.^2 .+ Vy_ref.^2)^0.5
            l_V_loc = Flux.Losses.mse(Vx_pred, inn1(Vx_ref); agg=mean) + Flux.Losses.mse(Vy_pred, inn1(Vy_ref); agg=mean)
            l_V += normVref^(-1) * l_V_loc
        else
            # Ice thickness
            if !isnothing(H_ref)
                l_H += Flux.Losses.mse(H[H_ref .!= 0.0], H_ref[H_ref.!= 0.0]; agg=mean) 
            end
            # Ice surface velocities
            l_V += Flux.Losses.mse(V_pred[V_ref .!= 0.0], V_ref[V_ref.!= 0.0]; agg=mean)
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

    println("Loss computed: $l_tot")

    return l_tot
    end # let
end

function predict_iceflow!(θ, simulation::FunctionalInversion, batch_ids::Vector{I}) where {I <: Integer}
    # Train UDE in parallel
    simulation.results = pmap((batch_id) -> batch_iceflow_UDE(θ, simulation, batch_id), batch_ids)
    println("All batches finished")
end

function batch_iceflow_UDE(θ, simulation::FunctionalInversion, batch_id::I) where {I <: Integer}
    
    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[batch_id]

    # glacier_id = isnothing(glacier.gdir) ? "unnamed" : glacier.rgi_id
    # println("Processing glacier: ", glacier_id)
    
    # Initialize glacier ice flow model
    initialize_iceflow_model(model.iceflow[batch_id], batch_id, glacier, params)

    params.solver.tstops =  @ignore_derivatives Huginn.define_callback_steps(params.simulation.tspan, params.solver.step)
    stop_condition(u,t,integrator) = Sleipnir.stop_condition_tstops(u,t,integrator, params.solver.tstops) #closure
    function action!(integrator)
        if params.simulation.use_MB 
            # Compute mass balance
            @ignore_derivatives begin 
                MB_timestep!(model, glacier, params.solver.step, integrator.t; batch_id = batch_id)
                apply_MB_mask!(integrator.u, glacier, model.iceflow[batch_id])
            end
        end
        # Apply parametrization
        apply_UDE_parametrization!(θ, simulation, integrator, batch_id)
    end
    
    cb_MB = DiscreteCallback(stop_condition, action!)

    # Run iceflow UDE for this glacier
    du = params.simulation.use_iceflow ? Huginn.SIA2D : Huginn.noSIA2D
    iceflow_sol = simulate_iceflow_UDE!(θ, simulation, model, params, cb_MB, batch_id; du = du)

    println("simulation finished for $batch_id")

    # Update simulation results
    results =  Sleipnir.create_results(simulation, batch_id, iceflow_sol, nothing; light=true, batch_id = batch_id)

    println("Batch $batch_id finished!")

    return results
end


"""
function simulate_iceflow_UDE!(
    θ,
    simulation::SIM, 
    model::Sleipnir.Model, 
    params::Sleipnir.Parameters, 
    cb::DiscreteCallback,
    batch_id::I; 
    du = Huginn.SIA2D) where {I <: Integer, SIM <: Simulation}

Make forward simulation of the iceflow UDE determined in `du`.
"""
function simulate_iceflow_UDE!(
    θ,
    simulation::SIM, 
    model::Sleipnir.Model, 
    params::Sleipnir.Parameters, 
    cb::DiscreteCallback,
    batch_id::I; 
    du = Huginn.SIA2D) where {I <: Integer, SIM <: Simulation}

    # TODO: make this more general
    apply_UDE_parametrization!(θ, simulation, nothing, batch_id)
    SIA2D_UDE_closure(H, θ, t) = SIA2D_UDE(H, θ, t, simulation, batch_id)

    iceflow_prob = ODEProblem(SIA2D_UDE_closure, model.iceflow[batch_id].H, params.simulation.tspan, tstops=params.solver.tstops, θ)
    iceflow_sol = solve(iceflow_prob, 
                        params.solver.solver, 
                        callback=cb,
                        tstops=params.solver.tstops, 
                        u0=model.iceflow[batch_id].H₀, 
                        p=θ,
                        sensealg=params.UDE.sensealg,
                        reltol=params.solver.reltol, 
                        save_everystep=false,  
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

function apply_UDE_parametrization!(θ, simulation::FunctionalInversion, integrator, batch_id::I) where {I <: Integer}
    # We load the ML model with the parameters
    U = simulation.model.machine_learning.NN_f(θ)
    # We generate the ML parametrization based on the target
    if simulation.parameters.UDE.target == "A"
        A = predict_A̅(U, [mean(simulation.glaciers[batch_id].climate.longterm_temps)])[1]
        simulation.model.iceflow[batch_id].A[] = A
    # elseif simulation.parameters.UDE.target == "D"
    #     parametrization = U()
    end
end

# Wrapper to pass a parametrization to the SIA2D
function SIA2D_UDE(H::Matrix{R}, θ, t::R, simulation::SIM, batch_id::I) where {R <: Real, I <: Integer, SIM <: Simulation}
    return Huginn.SIA2D(H, simulation, t; batch_id = batch_id)
end


callback_plots_A = function (θ, l, simulation) # callback function to observe training

    @ignore_derivatives begin
        
    update_training_state!(simulation, l)

    avg_temps = Float64[mean(simulation.glaciers[i].climate.longterm_temps) for i in 1:length(simulation.glaciers)]
    p = sortperm(avg_temps)
    avg_temps = avg_temps[p]
    # We load the ML model with the parameters
    U = simulation.model.machine_learning.NN_f(θ)
    pred_A = predict_A̅(U, collect(-23.0:1.0:0.0)')
    pred_A = Float64[pred_A...] # flatten
    true_A = A_fake(avg_temps, true)

    yticks = collect(0.0:2e-17:8e-17)

    training_path = joinpath(simulation.parameters.simulation.working_dir, "training")

    Plots.scatter(avg_temps, true_A, label="True A", c=:lightsteelblue2)
    plot_epoch = Plots.plot!(-23:1:0, pred_A, label="Predicted A", 
                        xlabel="Long-term air temperature (°C)", yticks=yticks,
                        ylabel="A", ylims=(0.0, simulation.parameters.physical.maxA), lw = 3, c=:dodgerblue4,
                        legend=:topleft)
    if !isdir(joinpath(training_path,"png")) || !isdir(joinpath(training_path,"pdf"))
        mkpath(joinpath(training_path,"png"))
        mkpath(joinpath(training_path,"pdf"))
    end
    # Plots.savefig(plot_epoch,joinpath(root_plots,"training","epoch$current_epoch.svg"))
    Plots.savefig(plot_epoch,joinpath(training_path,"png","epoch$(simulation.parameters.hyper.current_epoch).png"))
    Plots.savefig(plot_epoch,joinpath(training_path,"pdf","epoch$(simulation.parameters.hyper.current_epoch).pdf"))

    plot_loss = Plots.plot(simulation.parameters.hyper.loss_history, label="", xlabel="Epoch", yaxis=:log10,
                ylabel="Loss (V)", lw = 3, c=:darkslategray3)
    Plots.savefig(plot_loss,joinpath(training_path,"png","loss$(simulation.parameters.hyper.current_epoch).png"))
    Plots.savefig(plot_loss,joinpath(training_path,"pdf","loss$(simulation.parameters.hyper.current_epoch).pdf"))

    end #@ignore_derivatives 

    return false
end

