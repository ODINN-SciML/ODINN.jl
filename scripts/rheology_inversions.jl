## IMPORTANT: run this in the REPL before using ODINN! 
## Set up Python environment
# global ENV["PYTHON"] = "/home/jovyan/.conda/envs/oggm_env/bin/python3.9" # same as "which python" 
# import Pkg; Pkg.build("PyCall")
# exit()

import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise # very important!
using ODINN
using Plots
using Optim, OptimizationOptimJL
import OptimizationOptimisers.Adam
using Infiltrator
using Distributed
using JLD2
# using AbbreviatedStackTraces

# Activate to avoid GKS backend Plot issues
ENV["GKSwstype"]="nul"

processes = 10
# We enable multiprocessing
ODINN.enable_multiprocessing(processes)
# Flags
ODINN.set_use_MB(false)
ODINN.make_plots(true)    
# UDE training
ODINN.set_train(true)    # Train UDE
ODINN.set_retrain(false) # Re-use previous NN weights to continue training

function run()

    tspan = (2017, 2018) # period in years for simulation

    # Configure OGGM settings in all workers
    working_dir = joinpath(homedir(), "Python/OGGM_data")
    oggm_config(working_dir)    

    gtd_file, rgi_ids = ODINN.get_glathida_path_and_IDs()
    rgi_ids = rgi_ids[1:10] # filter for tests

    #######################################################################################################
    #############################         Train inversions         ########################################
    #######################################################################################################

    # Train iceflow UDE in parallel
    epochs = 50
    # optimizer = BFGS(initial_stepnorm=0.02f0)
    optimizer = Adam(0.001)
    train_settings = (BFGS(initial_stepnorm=0.05), epochs) # optimizer, epochs
    train_settings = (optimizer, epochs) # optimizer, epochs

    # Choose between "D" for diffusivity and "A" for Glen's coefficient
    @time rheology_trained = train_iceflow_inversion(rgi_ids, gtd_file, tspan, train_settings; target="D")
    θ_trained = rheology_trained.minimizer

    # Save trained NN weights
    println("Saving NN weights...")
    jldsave(joinpath(ODINN.root_dir, "data/trained_inv_weights.jld2"); θ_trained, ODINN.current_epoch)

    return rheology_trained

end

# Run main
rheology_trained = run()
