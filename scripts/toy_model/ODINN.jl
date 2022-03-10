import Pkg
cd(@__DIR__)
Pkg.activate(dirname(Base.current_project()))
Pkg.precompile()

## Environment and packages
using Distributed
using ProgressMeter
const processes = 16
if nprocs() < processes
    addprocs(processes - nprocs(); exeflags="--project")
end

println("Number of cores: ", nprocs())
println("Number of workers: ", nworkers())

################################################
############  PYTHON ENVIRONMENT  ##############
################################################

## Set up Python environment
# Choose own Python environment with OGGM's installation
# Use same path as "which python" in shell
ENV["PYTHON"] = "/home/jovyan/.conda/envs/oggm_env/bin/python3.9" # path in JupyterHub
Pkg.build("PyCall") 

@everywhere begin
using PyCall
using PyPlot # needed for Matplotlib plots

# Import OGGM sub-libraries in Julia
netCDF4 = pyimport("netCDF4")
cfg = pyimport("oggm.cfg")
utils = pyimport("oggm.utils")
workflow = pyimport("oggm.workflow")
tasks = pyimport("oggm.tasks")
graphics = pyimport("oggm.graphics")
bedtopo = pyimport("oggm.shop.bedtopo")
MBsandbox = pyimport("MBsandbox.mbmod_daily_oneflowline")

# Essential Python libraries
np = pyimport("numpy")
xr = pyimport("xarray")
# matplotlib = pyimport("matplotlib")
# matplotlib.use("Qt5Agg") 
end # @everywhere

###############################################
############  JULIA ENVIRONMENT  ##############
###############################################

@everywhere begin 
    import Pkg
    Pkg.activate(dirname(Base.current_project()))
end

@everywhere begin 
using Statistics
using LinearAlgebra
using Random
using Polynomials
using HDF5  
using JLD2
using OrdinaryDiffEq
using DiffEqFlux
using Zygote: @ignore
using Flux
using Tullio
using RecursiveArrayTools
using Infiltrator
using Plots
using ProgressMeter
using Dates # to provide correct Julian time slices 
using PyCall
using ParallelDataTransfer

###############################################
#############    PARAMETERS     ###############
###############################################

include("helpers/parameters.jl")

###############################################
#############  ODINN LIBRARIES  ###############
###############################################

cd(@__DIR__)
root_dir = dirname(Base.current_project())

### Climate data processing  ###
include("helpers/climate.jl")
### OGGM configuration settings  ###
include("helpers/oggm.jl")
end # @everywhere

### Iceflow modelling functions  ###
# (includes utils.jl as well)
include("helpers/iceflow.jl")

function main()
    # Configure OGGM settings in all workers
    @everywhere oggm_config()

    ###############################################################
    ###########################  MAIN #############################
    ###############################################################

    # Defining glaciers to be modelled with RGI IDs
    # RGI60-11.03638 # Argentière glacier (European Alps)
    # RGI60-11.01450 # Aletsch glacier (European Alps)
    # RGI60-08.00213 # Storglaciaren (Scandinavia)
    # RGI60-02.05098 # Peyto Glacier
    # RGI60-01.01104 # Lemon Creek Glacier (Alaska)
    # RGI60-01.09162 # Wolverine Glacier (Alaska)
    # RGI60-01.00570 # Gulkana Glacier (Alaska)
    # RGI60-07.00274 # Edvardbreen (Svalbard)
    # RGI60-07.01323 # Biskayerfonna (Svalbard)
    # RGI60-03.04207 # Canadian Arctic
    # RGI60-03.03533 # Canadian Arctic
    # RGI60-01.17316 # Twaharpies Glacier (Alaska)
    rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", 
                "RGI60-02.05098", "RGI60-01.01104", "RGI60-01.09162", "RGI60-01.00570",                	
                "RGI60-07.00274", "RGI60-07.01323", "RGI60-03.04207", "RGI60-03.03533", "RGI60-01.17316"]

    ### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
    gdirs = init_gdirs(rgi_ids)

    #########################################
    ###########  CLIMATE DATA  ##############
    #########################################

    # Process climate data for glaciers
    gdirs_climate = get_gdirs_with_climate(gdirs, true)

    # Run forward model for selected glaciers
    if create_ref_dataset 
        println("Generating reference dataset for training...")
    
        # Compute reference dataset in parallel
        @everywhere solver = Ralston()
        H_refs, V̄x_refs, V̄y_refs = @time generate_ref_dataset(gdirs_climate)
            
        println("Saving reference data")
        jldsave(joinpath(root_dir, "data/PDE_refs.jld2"); H_refs, V̄x_refs, V̄y_refs)
    end

    # Load stored PDE reference datasets
    PDE_refs = load(joinpath(root_dir, "data/PDE_refs.jld2"))

    #######################################################################################################
    #############################             Train UDE            ########################################
    #######################################################################################################

    # Training setup
    global current_epoch = 1

    cd(@__DIR__)
    global root_plots = cd(pwd, "../../plots")
    # Train iceflow UDE in parallel
    # First train with ADAM to move the parameters into a favourable space
    @everywhere solver = ROCK4()
    train_settings = (ADAM(0.05), 10) # optimizer, epochs
    iceflow_trained, UA = @time train_iceflow_UDE(gdirs_climate, train_settings, PDE_refs)
    θ_trained = iceflow_trained.minimizer

    # Continue training with a smaller learning rate
    # train_settings = (ADAM(0.001), 20) # optimizer, epochs
    # iceflow_trained = @time train_iceflow_UDE(H₀, UA, θ, train_settings, H_refs, temp_series)
    # θ_trained = iceflow_trained.minimizer

    # Continue training with BFGS
    # train_settings = (BFGS(initial_stepnorm=0.02f0), 20) # optimizer, epochs
    train_settings = (ADAM(0.002), 20) # optimizer, epochs
    iceflow_trained, UA = @time train_iceflow_UDE(gdirs_climate, train_settings, PDE_refs, θ_trained) # retrain
    θ_trained = iceflow_trained.minimizer

    # Save trained NN weights
    save(joinpath(root_dir, "data/trained_weights.jld"), "θ_trained", θ_trained)

    ##########################################
    ####  Plot the final trained model  ######
    ##########################################
    data_range = -20.0:0.0
    pred_A = predict_A̅(UA, θ_trained, collect(data_range)')
    pred_A = [pred_A...] # flatten
    true_A = A_fake(data_range) 

    Plots.scatter(true_A, label="True A")
    train_final = Plots.plot!(pred_A, label="Predicted A")
    Plots.savefig(train_final,joinpath(root_plots,"training","final_model.png"))
end

# Run main
main()
