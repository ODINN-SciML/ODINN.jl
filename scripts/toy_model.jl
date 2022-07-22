## IMPORTANT: run this in the REPL before using ODINN! 
## Set up Python environment
# global ENV["PYTHON"] = "/home/jovyan/.conda/envs/oggm_env/bin/python3.9" # same as "which python" 
# import Pkg; Pkg.build("PyCall")
# exit()

import Pkg
Pkg.activate(dirname(Base.current_project()))

using ODINN
using Optim, OptimizationOptimJL
import OptimizationOptimisers.Adam
using OrdinaryDiffEq
using Plots
using Infiltrator
using Distributed
using JLD2
using Statistics  
using AbbreviatedStackTraces

create_ref_dataset = false          # Run reference PDE to generate reference dataset
add_MB = true
retrain = false                     # Re-use previous NN weights to continue training

tspan = (0.0, 5.0) # period in years for simulation
processes = 16
# We enable multiprocessing
ODINN.enable_multiprocessing(processes)
 
function run()
    # Configure OGGM settings in all workers
    working_dir = joinpath(homedir(), "Python/OGGM_data")
    oggm_config(working_dir)

    # Defining glaciers to be modelled with RGI IDs
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", "RGI60-04.04351", "RGI60-01.02170",
                "RGI60-02.05098", "RGI60-01.01104", "RGI60-01.09162", "RGI60-01.00570", "RGI60-04.07051",                	
                "RGI60-07.00274", "RGI60-07.01323", "RGI60-03.04207", "RGI60-03.03533", "RGI60-01.17316"]

    ### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
    gdirs = init_gdirs(rgi_ids, force=false)

    #########################################
    ###########  CLIMATE DATA  ##############
    #########################################

    # Process climate data for glaciers
    gdirs_climate = get_gdirs_with_climate(gdirs, tspan, overwrite=false, plot=true)
    # Generate random mass balance series for toy model
    if add_MB
        random_MB = generate_random_MB(gdirs_climate, tspan)
    else
        random_MB = nothing
    end

    # Run forward model for selected glaciers
    if create_ref_dataset 
        println("Generating reference dataset for training...")
    
        # Compute reference dataset in parallel
        H_refs, V̄x_refs, V̄y_refs, S_refs, B_refs = @time generate_ref_dataset(gdirs_climate, tspan; random_MB=random_MB)

        println("Saving reference data")
        jldsave(joinpath(ODINN.root_dir, "data/PDE_refs.jld2"); H_refs, V̄x_refs, V̄y_refs, S_refs, B_refs)
    end

    # Load stored PDE reference datasets
    PDE_refs = load(joinpath(ODINN.root_dir, "data/PDE_refs.jld2"))
    # Plot training dataset of glaciers
    plot_glacier_dataset(gdirs_climate, PDE_refs)

    #######################################################################################################
    #############################             Train UDE            ########################################
    #######################################################################################################

    # Train iceflow UDE in parallel
    # First train with ADAM to move the parameters into a favourable space
    n_ADAM = 5
    n_BFGS = 80
    if retrain
        println("Retraining from previous NN weights...")
        trained_weights = load(joinpath(ODINN.root_dir, "data/trained_weights.jld2"))
        current_epoch = trained_weights["current_epoch"]
        θ_trained = trained_weights["θ_trained"]
        # train_settings = (BFGS(initial_stepnorm=0.05), 20) # optimizer, epochs
        train_settings = (BFGS(initial_stepnorm=0.02f0), n_BFGS) # optimizer, epochs
        iceflow_trained, UA_f = @time train_iceflow_UDE(gdirs_climate, tspan, train_settings, PDE_refs, θ_trained) # retrain
        θ_trained = iceflow_trained.minimizer

        # Save trained NN weights
        println("Saving NN weights...")
        jldsave(joinpath(ODINN.root_dir, "data/trained_weights.jld2"); θ_trained, current_epoch)
    else
        current_epoch = 1

        println("Training from scratch...")
        train_settings = (Adam(0.005), n_ADAM) # optimizer, epochs
        iceflow_trained, UA_f = @time train_iceflow_UDE(gdirs_climate, tspan, train_settings, PDE_refs)
        θ_trained = iceflow_trained.minimizer
        println("Saving NN weights...")
        jldsave(joinpath(ODINN.root_dir, "data/trained_weights.jld2"); θ_trained, current_epoch)

        # Continue training with BFGS
        #current_epoch = n_ADAM + 1
        train_settings = (BFGS(initial_stepnorm=0.02f0), n_BFGS) # optimizer, epochs
        iceflow_trained, UA_f = @time train_iceflow_UDE(gdirs_climate, tspan, train_settings, PDE_refs, θ_trained) # retrain
        θ_trained = iceflow_trained.minimizer
        # Save trained NN weights
        println("Saving NN weights...")
        jldsave(joinpath(ODINN.root_dir, "data/trained_weights.jld2"); θ_trained, current_epoch)

    end
end

# Run main
run()
