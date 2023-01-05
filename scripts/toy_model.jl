## IMPORTANT: run this in the REPL before using ODINN! 
## Set up Python environment
# global ENV["PYTHON"] = "/home/jovyan/.conda/envs/oggm_env/bin/python3.9" # same as "which python" 
# import Pkg; Pkg.build("PyCall")
# exit()

import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise
using ODINN
using Optim, OptimizationOptimJL
import OptimizationOptimisers.Adam
using OrdinaryDiffEq
using Plots
using Infiltrator
using Distributed
using JLD2
using Statistics  
# using AbbreviatedStackTraces

processes = 14
# Enable multiprocessing
ODINN.enable_multiprocessing(processes)
# Flags
ODINN.set_use_MB(false)
ODINN.make_plots(true)    
# Spin up 
ODINN.set_run_spinup(false) # Run the spin-up random_MB = generate_random_MB(gdirs_climate, tspan; plot=false)n
ODINN.set_use_spinup(false) # Use the updated spinup
# Reference simulations
ODINN.set_create_ref_dataset(true) # Generate reference data for UDE training
# UDE training
ODINN.set_train(true)    # Train UDE
ODINN.set_retrain(false) # Re-use previous NN weights to continue training

tspan = (2010.0, 2015.0) # period in years for simulation
 
function run()
    # Configure OGGM settings in all workers
    working_dir = joinpath(homedir(), "Python/OGGM_data")
    oggm_config(working_dir)

    # Defining glaciers to be modelled with RGI IDs
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", "RGI60-04.04351", "RGI60-01.02170",
                "RGI60-02.05098", "RGI60-01.01104", "RGI60-01.09162", "RGI60-01.00570", "RGI60-04.07051",                	
                "RGI60-07.00274", "RGI60-07.01323", "RGI60-03.04207", "RGI60-03.03533", "RGI60-01.17316"]

    ### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
    gdirs = init_gdirs(rgi_ids[4:7])

    #########################################
    ###########  CLIMATE DATA  ##############
    #########################################

    # Process climate data for glaciers
    gdirs_climate, gdirs_climate_batches = get_gdirs_with_climate(gdirs, tspan, overwrite=false, plot=false)
    # Generate random mass balance series for toy model
    if ODINN.use_MB[]
        random_MB = generate_random_MB(gdirs_climate, tspan; plot=false)
    else
        random_MB = nothing
    end

    if ODINN.run_spinup[]
        println("Spin up run to stabilize initial conditions...")
        @time spinup(gdirs_climate, tspan; random_MB=random_MB)
    end

    # Run forward model for selected glaciers
    if ODINN.create_ref_dataset[]
        println("Generating reference dataset for training...")
    
        # Compute reference dataset in parallel
        gdir_refs = @time generate_ref_dataset(gdirs_climate, tspan; random_MB=random_MB)

        println("Saving reference data")
        jldsave(joinpath(ODINN.root_dir, "data/gdir_refs.jld2"); gdir_refs)
    end

    # Load stored PDE reference datasets
    gdir_refs = load(joinpath(ODINN.root_dir, "data/gdir_refs.jld2"))["gdir_refs"]

    # Plot training dataset of glaciers
    # plot_glacier_dataset(gdirs_climate, gdir_refs, random_MB)

    #######################################################################################################
    #############################             Train UDE            ########################################
    #######################################################################################################

    if ODINN.train[]
        # Train iceflow UDE in parallel
        # First train with ADAM to move the parameters into a favourable space
        n_ADAM = 5
        n_BFGS = 40
        if ODINN.retrain[]
            println("Retraining from previous NN weights...")
            trained_weights = load(joinpath(ODINN.root_dir, "data/trained_weights.jld2"))
            ODINN.set_current_epoch(trained_weights["current_epoch"])
            θ_trained = trained_weights["θ_trained"]
            # train_settings = (BFGS(initial_stepnorm=0.05), 20) # optimizer, epochs
            batch_size = length(gdir_refs)
            train_settings = (BFGS(initial_stepnorm=0.02), n_BFGS, batch_size) # optimizer, epochs, batch_size
            iceflow_trained, UA_f, loss_history = @time train_iceflow_UDE(gdirs_climate, gdirs_climate_batches, 
                                                                        tspan, train_settings, gdir_refs, θ_trained; 
                                                                        random_MB=random_MB) # retrain
            θ_trained = iceflow_trained.minimizer

            # Save trained NN weights
            println("Saving NN weights...")
            jldsave(joinpath(ODINN.root_dir, "data/trained_weights.jld2"); θ_trained, ODINN.current_epoch)
        else
            # Reset epoch counter
            ODINN.reset_epochs()

            println("Training from scratch...")
            # train_settings = (Adam(0.005), n_ADAM) # optimizer, epochs
            # iceflow_trained, UA_f = @time train_iceflow_UDE(gdirs_climate, tspan, train_settings, PDE_refs)
            # θ_trained = iceflow_trained.minimizer
            # println("Saving NN weights...")
            # jldsave(joinpath(ODINN.root_dir, "data/trained_weights.jld2"); θ_trained, current_epoch)

            # Continue training with BFGS
            #current_epoch = n_ADAM + 1
            batch_size = length(gdir_refs)
            optimizer = BFGS(initial_stepnorm=0.001)
            # optimizer = Adam(0.001)
            train_settings = (optimizer, n_BFGS, batch_size) # optimizer, epochs, batch_size
            iceflow_trained, UA_f, loss_history = @time train_iceflow_UDE(gdirs_climate, gdirs_climate_batches, 
                                                                        tspan, train_settings, gdir_refs; 
                                                                        random_MB=random_MB) 
            # iceflow_trained, UA_f = @time train_iceflow_UDE(gdirs_climate, tspan, train_settings, PDE_refs, θ_trained) # retrain
            θ_trained = iceflow_trained.minimizer
            # Save loss loss_history
            jldsave(joinpath(ODINN.root_dir, "data/loss_history.jld2"); loss_history)
            # Save trained NN weights
            println("Saving NN weights...")
            jldsave(joinpath(ODINN.root_dir, "data/trained_weights.jld2"); θ_trained, ODINN.current_epoch)
        end
    end
end

# Run main
run()
