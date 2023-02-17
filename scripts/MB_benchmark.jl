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

using Dates

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

    ## Redesigning the climate and MB series workflow
    ##  Raw skeleton for function to retrieve raw climate for all gdirs
    @timeit to "process raw climate" begin
    for gdir in gdirs # make this a pmap in the function
        println("Getting raw  climate for: ", gdir.rgi_id)
        # Get raw climate data for gdir
        tspan_date = ODINN.partial_year(Day, tspan[1]):Day(1):ODINN.partial_year(Day, tspan[2])
        climate =  @timeit to "get raw climate" ODINN.get_raw_climate_data(gdir)
        # Make sure the desired period is covered by the climate data
        period = ODINN.trim_period(tspan_date, climate) 
        if any((climate.time[1].dt.date.data[1] <= period[1]) & any(climate.time[end].dt.date.data[1] >= period[end]))
            climate = climate.sel(time=period) # Crop desired time period
        else
            @warn "No overlapping period available between climate and target data!" 
        end
        # Save gdir climate on disk 
        @eval ODINN $climate.to_netcdf(joinpath($gdir.dir, "raw_climate.nc"))
    end
    end
    @timeit to "compute MB one step" begin
    # Raw skeleton for function to get MB for a given timestepping
    step = 1.0/12.0
    # We define a temperature-index model with a single DDF
    ti_mb_model = ODINN.TI_model_1(DDF=5.0, acc_factor=1.2)
    for gdir in gdirs
        climate = ODINN.xr.open_dataset(joinpath(gdir.dir, "raw_climate.nc")) # load only once at the beginning
        dem = ODINN.xr.open_rasterio(gdir.get_filepath("dem"))
        # for time loop
        t = 2012.3567 # time for testing
        # First we get the dates of the current time and the previous step
        period = ODINN.partial_year(Day, t - step):Day(1):ODINN.partial_year(Day, t)
        climate_step = climate.sel(time=period) # Crop desired time period
        climate_step = ODINN.get_cumulative_climate(climate_step)
        # Convert climate dataset to 2D based on the glacier's DEM
        climate_2D_step = ODINN.downscale_2D_climate(climate_step, dem)
        MB = ODINN.compute_MB(ti_mb_model, climate_2D_step)
        @show maximum(MB)
        @show minimum(MB)
        @show mean(MB)
    end
    end

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



