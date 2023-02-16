################################################
############  PYTHON ENVIRONMENT  ##############
################################################

import Pkg 
Pkg.activate(dirname(Base.current_project()))

using Distributed

@everywhere begin

using Revise
# using AbbreviatedStackTraces
using ODINN
using OrdinaryDiffEq, ODEInterfaceDiffEq, Sundials
using Optim, Optimization, OptimizationOptimJL, SciMLSensitivity
import OptimizationOptimisers.Adam
using JLD2
using BenchmarkTools, TimerOutputs
using Infiltrator

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

end # @everywhere

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

# Create a TimerOutput, this is the main type that keeps track of everything.
const to = TimerOutput()

# We enable multiprocessing
# processes = 15
# ODINN.enable_multiprocessing(processes)

###############################################################
###########################  MAIN #############################
###############################################################


function run_benchmark()
    
    # Flags
    @timeit to "setters" begin
    ODINN.set_use_MB(true) 
    # Spin up and reference simulations
    ODINN.set_run_spinup(false) # Run the spin-up simulation
    ODINN.set_use_spinup(false) # Use the updated spinup
    ODINN.set_create_ref_dataset(false) # Generate reference data for UDE training
    # UDE training
    ODINN.set_train(true)    # Train UDE 

    ODINN.set_optimization_method("AD+Diff")
    end

    tspan = (2010.0,2015.0) # period in years for simulation

    # Configure OGGM settings in all workers
    # Use a separate working dir to avoid conflicts with other simulations
    working_dir = joinpath(homedir(), "Python/OGGM_memory_benchmark")
    @timeit to "oggm config" oggm_config(working_dir)

    # Defining glaciers to be modelled with RGI IDs
    # RGI60-11.03638 # Argentière glacier
    # RGI60-11.01450 # Aletsch glacier
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]

    ### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
    gdirs = @timeit to "init_gdirs" init_gdirs(rgi_ids)

    #########################################
    ###########  CLIMATE DATA  ##############
    #########################################

    # Process climate data for glaciers
    gdirs_climate, gdirs_climate_batches = @timeit to "climate" get_gdirs_with_climate(gdirs, tspan, overwrite=false, plot=false)

    @show to

    @infiltrate

    # Generate random mass balance series for toy model
    if ODINN.use_MB[]
        random_MB = generate_random_MB(gdirs_climate, tspan; plot=false)
    else
        random_MB = nothing
    end
    # Run forward model for selected glaciers
    if ODINN.create_ref_dataset
        println("Generating reference dataset for training...")
        solver = RDPK3Sp35()
        # solver = Ralston()
        # Compute reference dataset in parallel
        gdir_refs = @time generate_ref_dataset(gdirs_climate, tspan; solver=solver, random_MB=random_MB)

        println("Saving reference benchmark data")
        jldsave(joinpath(ODINN.root_dir, "data/PDE_refs_benchmark.jld2"); gdir_refs)
    end

end

run_benchmark()



