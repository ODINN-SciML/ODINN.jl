"""
    callback_plots_A(θ, l, simulation)

Callback function to generate plots during training.
"""
function callback_plots_A(θ, l, simulation)

    @ignore_derivatives begin

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
    callback_diagnosis(θ, l, simulation)

Callback function to track and diagose training. It includes print and updates in simulation::Simulation.
"""
function callback_diagnosis(θ, l, simulation)

    # See if we want to change this or leave it
    # update_training_state!(simulation, l)

    push!(simulation.stats.losses, l)
    step = 1
    if length(simulation.stats.losses) % step == 0
        if length(simulation.stats.losses) == 1
            improvement = nothing
        else
            improvement = (l - simulation.stats.losses[end-step]) / simulation.stats.losses[end-step]
        end
        printProgressLoss(length(simulation.stats.losses), simulation.stats.niter, l, improvement)
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