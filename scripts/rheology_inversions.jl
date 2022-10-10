## IMPORTANT: run this in the REPL before using ODINN! 
## Set up Python environment
# global ENV["PYTHON"] = "/home/jovyan/.conda/envs/oggm_env/bin/python3.9" # same as "which python" 
# import Pkg; Pkg.build("PyCall")
# exit()

import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise # very important!
using ODINN
using Optim, OptimizationOptimJL
import OptimizationOptimisers.Adam
using OrdinaryDiffEq
using Plots
using Infiltrator
using Distributed
using JLD2, HDF5, Downloads
using Statistics  
# using AbbreviatedStackTraces

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

    # Download all data from Glathida
    gtd_file = Downloads.download("https://cluster.klima.uni-bremen.de/~oggm/glathida/glathida-v3.1.0/data/TTT_per_rgi_id.h5")
    glathida = h5open(gtd_file, "r")
    # Retrieve RGI IDs with Glathida data
    rgi_ids = keys(glathida)
    # Delete Greenland and Antarctic glaciers, for now
    deleteat!(rgi_ids, findall(x->x[begin:8]=="RGI60-05",rgi_ids))
    deleteat!(rgi_ids, findall(x->x[begin:8]=="RGI60-19",rgi_ids))  

    rgi_ids = rgi_ids[1:10] # filter for tests

    gtd_df = pd.read_hdf(gtd_file, key=rgi_ids[1])

    # TODO: build matrix version of each Glathida ice thickness data for each glacier

    @infiltrate

    # Configure OGGM settings in all workers
    working_dir = joinpath(homedir(), "Python/OGGM_data")
    oggm_config(working_dir)    

    #######################################################################################################
    #############################         Train inversions         ########################################
    #######################################################################################################

    # Train iceflow UDE in parallel
    n_BFGS = 40
    # train_settings = (BFGS(initial_stepnorm=0.05), 20) # optimizer, epochs
    train_settings = (BFGS(initial_stepnorm=0.02f0), n_BFGS) # optimizer, epochs

    @time train_iceflow_inversion(rgi_ids, glathida, tspan, train_settings)
    θ_trained = iceflow_trained.minimizer

    # Save trained NN weights
    println("Saving NN weights...")
    jldsave(joinpath(ODINN.root_dir, "data/trained_inv_weights.jld2"); θ_trained, ODINN.current_epoch)

end

# Run main
run()
