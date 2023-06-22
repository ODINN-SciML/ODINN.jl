

function run!(simulation::Simulation)

    @show simulation

end


"""
    stop_condition_tstops(u,t,integrator, tstops)  

Function that iterates through the tstops, with a closure including `tstops`
"""
function stop_condition_tstops(u,t,integrator, tstops) 
    t in tstops
end

callback_plots_A = function (θ, l, UA_f, longterm_temps, A_noise, training_path, batch_size, n_gdirs) # callback function to observe training
    # Update training status
    update_training_state(l, batch_size, n_gdirs)

    if current_minibatches == 0.0
        avg_temps = Float64[mean(longterm_temps[i]) for i in 1:length(longterm_temps)]
        p = sortperm(avg_temps)
        avg_temps = avg_temps[p]
        pred_A = predict_A̅(UA_f, θ, collect(-23.0:1.0:0.0)')
        pred_A = Float64[pred_A...] # flatten
        true_A = A_fake(avg_temps, A_noise[p], noise)

        yticks = collect(0.0:2e-17:8e-17)

        Plots.scatter(avg_temps, true_A, label="True A", c=:lightsteelblue2)
        plot_epoch = Plots.plot!(-23:1:0, pred_A, label="Predicted A", 
                            xlabel="Long-term air temperature (°C)", yticks=yticks,
                            ylabel="A", ylims=(0.0,maxA[]), lw = 3, c=:dodgerblue4,
                            legend=:topleft)
        if !isdir(joinpath(training_path,"png")) || !isdir(joinpath(training_path,"pdf"))
            mkpath(joinpath(training_path,"png"))
            mkpath(joinpath(training_path,"pdf"))
        end
        # Plots.savefig(plot_epoch,joinpath(root_plots,"training","epoch$current_epoch.svg"))
        Plots.savefig(plot_epoch,joinpath(training_path,"png","epoch$(current_epoch).png"))
        Plots.savefig(plot_epoch,joinpath(training_path,"pdf","epoch$(current_epoch).pdf"))

        plot_loss = Plots.plot(loss_history, label="", xlabel="Epoch", yaxis=:log10,
                    ylabel="Loss (V)", lw = 3, c=:darkslategray3)
        Plots.savefig(plot_loss,joinpath(training_path,"png","loss$(current_epoch).png"))
        Plots.savefig(plot_loss,joinpath(training_path,"pdf","loss$(current_epoch).pdf"))
    end

    return false
end