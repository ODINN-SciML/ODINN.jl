"""
    callback_plots_A(θ, l, simulation)

Callback function to generate plots during training.
"""
function callback_plots_A(θ, l, simulation)
    @assert inputs(simulation.model.iceflow.A)==_inputs_A_law_scalar ||
            inputs(simulation.model.iceflow.A)==_inputs_A_law_gridded "In order to use `callback_plots_A`, A law must be used and its inputs must be $(_inputs_A_law_scalar) or $(_inputs_A_law_gridded)"

    @ignore_derivatives begin
        t = simulation.parameters.simulation.tspan[1] # We need to provide t but it isn't important here
        res = map(glacier_idx -> T_A_Alaw(simulation, glacier_idx, θ, t), 1:length(simulation.glaciers))
        avg_temps, avg_temps = collect.(zip(res...))

        Tvec = collect(-23.0:1.0:0.0)
        pred_A = map(
            Ti -> eval_law(
                simulation.model.iceflow.A, simulation, 1, (; T = Ti), θ), Tvec)

        A_poly = Huginn.polyA_PatersonCuffey()
        true_A = A_poly.(avg_temps)

        yticks = collect(0.0:2e-17:8e-17)

        training_path = joinpath(simulation.parameters.simulation.working_dir, "training")

        fig_epoch = Figure()
        ax_epoch = Axis(fig_epoch[1, 1];
            xlabel = "Long-term air temperature (°C)", ylabel = "A",
            yticks = yticks,
            limits = (nothing, (0.0, simulation.parameters.physical.maxA)))
        scatter!(ax_epoch, avg_temps, true_A; label = "True A", color = :lightsteelblue2)
        lines!(ax_epoch, Tvec, pred_A; label = "Predicted A", linewidth = 3,
            color = :dodgerblue4)
        axislegend(ax_epoch; position = :lt)

        if !isdir(joinpath(training_path, "png")) || !isdir(joinpath(training_path, "pdf"))
            mkpath(joinpath(training_path, "png"))
            mkpath(joinpath(training_path, "pdf"))
        end
        CairoMakie.save(
            joinpath(training_path, "png",
                "epoch$(simulation.parameters.hyper.current_epoch).png"),
            fig_epoch)
        CairoMakie.save(
            joinpath(training_path, "pdf",
                "epoch$(simulation.parameters.hyper.current_epoch).pdf"),
            fig_epoch)

        fig_loss = Figure()
        ax_loss = Axis(fig_loss[1, 1]; xlabel = "Epoch", ylabel = "Loss (V)",
            yscale = log10)
        lines!(ax_loss, simulation.parameters.hyper.loss_history;
            linewidth = 3, color = :darkslategray3)
        CairoMakie.save(
            joinpath(training_path, "png",
                "loss$(simulation.parameters.hyper.current_epoch).png"),
            fig_loss)
        CairoMakie.save(
            joinpath(training_path, "pdf",
                "loss$(simulation.parameters.hyper.current_epoch).pdf"),
            fig_loss)
    end #@ignore_derivatives 

    return false
end

"""
    callback_diagnosis(θ, l, simulation; save::Bool = false, tbLogger::Union{<: TBLogger, Nothing} = nothing)

Callback function to track and diagose training.
It includes print and updates in simulation::Simulation.
It also logs training statistics with tensorboard if tbLogger is provided.
"""
function callback_diagnosis(θ, l, simulation; save::Bool = false,
        tbLogger::Union{<: TBLogger, Nothing} = nothing)
    currentDt = now()

    # See if we want to change this or leave it
    # update_training_state!(simulation, l)

    push!(simulation.results.stats.losses, l)
    push!(simulation.results.stats.θ_hist, θ.u)
    push!(simulation.results.stats.∇θ_hist, θ.grad)

    step = 1
    if length(simulation.results.stats.losses) % step == 0
        if length(simulation.results.stats.losses) == 1
            improvement = nothing
        else
            improvement = (l - simulation.results.stats.losses[end - step]) /
                          simulation.results.stats.losses[end - step]
        end
        printProgressLoss(length(simulation.results.stats.losses),
            simulation.results.stats.niter, l, improvement)
    end

    if !isnothing(tbLogger)
        iter = length(simulation.results.stats.losses) # Use this instead of θ.iter to be able to log optimizations with multiple optimizers in the same tensorboard run
        log_value(tbLogger, "train/loss", θ.objective, step = iter)
        if !isnothing(θ.grad)
            log_value(tbLogger, "train/norm_grad", norm(θ.grad), step = iter)
        end
        if !isnothing(θ.hess)
            log_value(tbLogger, "train/norm_hess", norm(θ.hess), step = iter)
        end
        if simulation.results.stats.lastCall != DateTime(0, 1, 1)
            time_per_iter = Millisecond(currentDt - simulation.results.stats.lastCall).value/1000
            log_value(tbLogger, "train/time_per_iter", time_per_iter, step = iter)
        end
        simulation.results.stats.lastCall = currentDt
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
            printstyled(@sprintf("%.2f %%", 100*improvement); color = :green)
        else
            printstyled(@sprintf("%.2f %%", 100*improvement); color = :red)
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
