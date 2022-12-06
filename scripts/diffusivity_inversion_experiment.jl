## IMPORTANT: run this in the REPL before using ODINN! 
## Set up Python environment
# global ENV["PYTHON"] = "/home/jovyan/.conda/envs/oggm_env/bin/python3.9" # same as "which python" 
# import Pkg; Pkg.build("PyCall")
# exit()

import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise # very important!
# using AbbreviatedStackTraces
using ODINN
using Plots
using Optim, OptimizationOptimJL
import OptimizationOptimisers.Adam
using Infiltrator
using Distributed
using JLD2

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

# ODINN settings
processes = 20
# We enable multiprocessing
ODINN.enable_multiprocessing(processes)
# Flags
ODINN.set_use_MB(true)
ODINN.make_plots(true)    
# Reference dataset
ODINN.set_create_ref_dataset(false) # Generate reference data for UDE training
# UDE training
ODINN.set_retrain(false) # Re-use previous NN weights to continue training
# Datasets for inversion
ODINN.set_ice_thickness_source("farinotti")

function run()

    tspan = (2017, 2018) # period in years for simulation

    # Configure OGGM settings in all workers
    working_dir = joinpath(homedir(), "Python/OGGM_data_diffusivity")
    oggm_config(working_dir)    

    # Defining glaciers to be modelled with RGI IDs
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", "RGI60-04.04351", "RGI60-01.02170",
                "RGI60-02.05098", "RGI60-01.01104", "RGI60-01.09162", "RGI60-01.00570", "RGI60-04.07051",                	
                "RGI60-07.00274", "RGI60-07.01323", "RGI60-03.04207", "RGI60-03.03533", "RGI60-01.17316", 
                "RGI60-07.01193", "RGI60-01.22174", "RGI60-14.07309", "RGI60-15.10261"]

    ### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
    gdirs = init_gdirs(rgi_ids)

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

    #########################################
    #########  REFERENCE DATASET  ###########
    #########################################

    # Run forward model for selected glaciers
    if ODINN.create_ref_dataset[]
        println("Generating reference dataset for training...")
    
        # Compute reference dataset in parallel
        gdir_refs = @time generate_ref_dataset(gdirs_climate, tspan; random_MB=random_MB)

        println("Saving reference data")
        jldsave(joinpath(ODINN.root_dir, "data/D_inv_experiment/gdir_refs.jld2"); gdir_refs)

    else
        gdir_refs = load(joinpath(ODINN.root_dir, "data/D_inv_experiment/gdir_refs.jld2"))["gdir_refs"]
    end

    # Plot training dataset of glaciers
    # plot_glacier_dataset(gdirs_climate, gdir_refs, random_MB)

    #######################################################################################################
    #############################         Train inversions         ########################################
    #######################################################################################################

    # Train iceflow UDE in parallel
    # Choose between "D" for diffusivity and "A" for Glen's coefficient
    if ODINN.retrain[]
        println("Retraining from previous NN weights...")
        trained_weights = load(joinpath(ODINN.root_dir, "data/trained_inv_weights.jld2"))
        θ_trained = trained_weights["θ_trained"]
        ODINN.set_current_epoch(trained_weights["current_epoch"])

        # Continue training with BFGS
        epochs_BFGS = 200
        batch_size = 4
        # optimizer = BFGS(initial_stepnorm=0.001f0)
        optimizer = Adam(0.01)
        train_settings = (optimizer, epochs_BFGS, batch_size) # optimizer, epochs, batch size
        @time rheology_trained = train_iceflow_inversion(rgi_ids, tspan, train_settings; 
                                                        gdirs_climate=gdirs_climate,
                                                        gdirs_climate_batches=gdirs_climate_batches, 
                                                        gdir_refs=gdir_refs, 
                                                        θ_trained=θ_trained, 
                                                        target="A")           
        θ_trained = rheology_trained.minimizer

        # Save trained NN weights
        println("Saving NN weights...")
        jldsave(joinpath(ODINN.root_dir, "data/trained_inv_weights.jld2"); θ_trained, ODINN.current_epoch)
    else
        epochs = 250
        batch_size = 10
        optimizer = BFGS(initial_stepnorm=0.001f0)
        # optimizer = LBFGS()
        # optimizer = Adam(0.003)
        train_settings = (optimizer, epochs, batch_size) # optimizer, epochs
    
        @time rheology_trained = train_iceflow_inversion(rgi_ids, tspan, train_settings; 
                                                        gdirs_climate=gdirs_climate,
                                                        gdirs_climate_batches=gdirs_climate_batches, 
                                                        gdir_refs=gdir_refs, 
                                                        target="A")  
        θ_trained = rheology_trained.minimizer

        # Save trained NN weights
        println("Saving NN weights...")
        jldsave(joinpath(ODINN.root_dir, "data/trained_inv_weights.jld2"); θ_trained, ODINN.current_epoch)
    end

    return rheology_trained

end

# Run main
rheology_trained = run()
