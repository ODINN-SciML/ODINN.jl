################################################
############  PYTHON ENVIRONMENT  ##############
################################################

import Pkg 
Pkg.activate(dirname(Base.current_project()))

using ODINN
using OrdinaryDiffEq, Optim, Optimization, SciMLSensitivity
import OptimizationOptimisers.Adam
using JLD2
using BenchmarkTools
using Infiltrator

create_ref_dataset = true  # Run reference PDE to generate reference dataset

tspan = (0.0,5.0) # period in years for simulation
processes = 3

# We enable multiprocessing
ODINN.enable_multiprocessing(processes)

###############################################################
###########################  MAIN #############################
###############################################################

function run_benchmark()

    # Configure OGGM settings in all workers
    # Use a separate working dir to avoid conflicts with other simulations
    working_dir = joinpath(homedir(), "Python/OGGM_data_benchmark")
    oggm_config(working_dir)

    # Defining glaciers to be modelled with RGI IDs
    # RGI60-11.03638 # Argentière glacier
    # RGI60-11.01450 # Aletsch glacier
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]

    ### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
    gdirs = init_gdirs(rgi_ids, force=false)

    glacier_filter = 1
    gdir = [gdirs[glacier_filter]]

    #########################################
    ###########  CLIMATE DATA  ##############
    #########################################

    # Process climate data for glaciers
    gdirs_climate = get_gdirs_with_climate(gdir, tspan, overwrite=false, plot=false)

    # Run forward model for selected glaciers
    if create_ref_dataset 
        println("Generating reference dataset for training...")

        # Compute reference dataset in parallel
        H_refs, V̄x_refs, V̄y_refs, S_refs, B_refs = @time generate_ref_dataset(gdirs_climate, tspan)

        println("Saving reference benchmark data")
        jldsave(joinpath(ODINN.root_dir, "data/PDE_refs_benchmark.jld2"); H_refs, V̄x_refs, V̄y_refs, S_refs, B_refs)
    end

    # Load stored PDE reference datasets
    PDE_refs = load(joinpath(ODINN.root_dir, "data/PDE_refs_benchmark.jld2"))

    #######################################################################################################
    #############################             Train UDE            ########################################
    #######################################################################################################

    θ_bm = load(joinpath(ODINN.root_dir, "data/benchmark_weights.jld"))["θ_benchmark"]

    # Solvers 
    # bsolver = Ralston()
    # bsolver = TRBDF2()
    bsolver = ROCK4()

    current_epoch = 1
    reltol = 10e-6
    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    train_settings = (Adam(0.005), 2) # optimizer, epochs
    UDE_settings = Dict("reltol"=>reltol,
                        "solver"=>bsolver,
                        "sensealg"=>sensealg)

    println("Benchmarking UDE settings: ", UDE_settings)
    @benchmark train_iceflow_UDE($gdirs_climate, $tspan, $train_settings, $PDE_refs, $θ_bm, $UDE_settings)

end

run_benchmark()