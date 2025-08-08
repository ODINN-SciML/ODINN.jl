"""
    callback_plots_A(θ, l, simulation)

Callback function to generate plots during training.
"""
function callback_plots_A(θ, l, simulation)
    @assert inputs(simulation.model.iceflow.A)==_inputs_A_law "In order to use `callback_plots_A`, A law must be used and its inputs must be $(_inputs_A_law)"

    @ignore_derivatives begin

    t = simulation.parameters.simulation.tspan[1] # We need to provide t but it isn't important here
    res = map(glacier_idx -> T_A_Alaw(simulation, glacier_idx, θ, t), 1:length(simulation.glaciers))
    avg_temps, avg_temps = collect.(zip(res...))

    Tvec = collect(-23.0:1.0:0.0)
    pred_A = map(Ti -> eval_law(simulation.model.iceflow.A, simulation, 1, (;T=Ti), θ), Tvec)

    A_poly = Huginn.polyA_PatersonCuffey()
    true_A = A_poly.(avg_temps)

    yticks = collect(0.0:2e-17:8e-17)

    training_path = joinpath(simulation.parameters.simulation.working_dir, "training")

    Plots.scatter(avg_temps, true_A, label="True A", c=:lightsteelblue2)
    plot_epoch = Plots.plot!(Tvec, pred_A, label="Predicted A",
                        xlabel="Long-term air temperature (°C)", yticks=yticks,
                        ylabel=:A, ylims=(0.0, simulation.parameters.physical.maxA), lw = 3, c=:dodgerblue4,
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

"""
    callback_diagnosis(θ, l, simulation; save::Bool = false, tbLogger::Union{<: TBLogger, Nothing} = nothing)

Callback function to track and diagose training.
It includes print and updates in simulation::Simulation.
It also logs training statistics with tensorboard if tbLogger is provided.
"""
function callback_diagnosis(θ, l, simulation; save::Bool = false, tbLogger::Union{<: TBLogger, Nothing} = nothing)
    currentDt = now()

    # See if we want to change this or leave it
    # update_training_state!(simulation, l)

    push!(simulation.stats.losses, l)
    push!(simulation.stats.θ_hist, θ.u)
    push!(simulation.stats.∇θ_hist, θ.grad)

    step = 1
    if length(simulation.stats.losses) % step == 0
        if length(simulation.stats.losses) == 1
            improvement = nothing
        else
            improvement = (l - simulation.stats.losses[end-step]) / simulation.stats.losses[end-step]
        end
        printProgressLoss(length(simulation.stats.losses), simulation.stats.niter, l, improvement)
    end

    if !isnothing(tbLogger)
        iter = length(simulation.stats.losses) # Use this instead of θ.iter to be able to log optimizations with multiple optimizers in the same tensorboard run
        log_value(tbLogger, "train/loss", θ.objective, step=iter)
        if !isnothing(θ.grad)
            log_value(tbLogger, "train/norm_grad", norm(θ.grad), step=iter)
        end
        if !isnothing(θ.hess)
            log_value(tbLogger, "train/norm_hess", norm(θ.hess), step=iter)
        end
        if simulation.stats.lastCall != DateTime(0,1,1)
            time_per_iter = Millisecond(currentDt - simulation.stats.lastCall).value/1000
            log_value(tbLogger, "train/time_per_iter", time_per_iter, step=iter)
        end
        simulation.stats.lastCall = currentDt
    end

    if save
        # Save state of intermediate simulation
        if !isnothing(Base.source_path)
            path = Base.dirname(Base.source_path())
        else
            path = @__DIR__
            println("Saving intermediate solution in $(path).")
        end
        ODINN.save_inversion_file!(θ, simulation; path = path, file_name = "_inversion_result.jld2")
    end

    return false
end

"""
    printProgressLoss(iter, total_iters, loss, improvement)

Print function to track training.
"""
function printProgressLoss(iter, total_iters, loss, improvement)
    print("Iteration: [")
    print(@sprintf("%5i", iter))
    print(" / ")
    print(@sprintf("%5i", total_iters))
    print("]     ")
    print("Loss:")
    print(@sprintf("%9.5e", loss))
    if !isnothing(improvement)
        print("     ")
        print("Improvement: ")
        if improvement <= 0
            printstyled(@sprintf("%.2f %%", 100*improvement); color=:green)
        else
            printstyled(@sprintf("%.2f %%", 100*improvement); color=:red)
        end
    end
    println("")
end


"""
    CallbackOptimizationSet(θ, l; callbacks)

Helper to combine callbacks for Optimization function. This executes the action of each callback.  
(equivalent to CallbackSet for DifferentialEquations.jl)
"""
function CallbackOptimizationSet(θ, l; callbacks)
    for cb in callbacks
        _ = cb(θ, l)
    end
    return false
end