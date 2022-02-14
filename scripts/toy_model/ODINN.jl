################################################
############  PYTHON ENVIRONMENT  ##############
################################################

import Pkg
Pkg.activate(dirname(Base.current_project()))
Pkg.precompile()

## Set up Python environment
# Choose own Python environment with OGGM's installation
# Use same path as "which python" in shell
ENV["PYTHON"] = "/home/jovyan/.conda/envs/oggm_env/bin/python3.9" # path in JupyterHub
Pkg.build("PyCall") 

using PyCall
using PyPlot # needed for Matplotlib plots

# Import OGGM sub-libraries in Julia
cfg = pyimport("oggm.cfg")
utils = pyimport("oggm.utils")
workflow = pyimport("oggm.workflow")
tasks = pyimport("oggm.tasks")
graphics = pyimport("oggm.graphics")
bedtopo = pyimport("oggm.shop.bedtopo")
# MBsandbox = pyimport("MBsandbox.mbmod_daily_oneflowline") # TODO: fix issue with Python version in Gemini HPC

# Essential Python libraries
np = pyimport("numpy")
xr = pyimport("xarray")
# matplotlib = pyimport("matplotlib")
# matplotlib.use("Qt5Agg") 

###############################################
############  JULIA ENVIRONMENT  ##############
###############################################

## Environment and packages
using Distributed
using ProgressMeter
const processes = 10

if nprocs() < processes
    addprocs(processes - nprocs(); exeflags="--project")
end

println("Number of cores: ", nprocs())
println("Number of workers: ", nworkers())

@everywhere begin 
    import Pkg
    Pkg.activate(dirname(Base.current_project()))
    Pkg.precompile()
end

@everywhere begin 
using Statistics
using LinearAlgebra
using Random
using HDF5  
using JLD2
using OrdinaryDiffEq
using DiffEqFlux
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
end # @everywhere

### Iceflow forward model  ###
# (includes utils.jl as well)
include("helpers/iceflow.jl")

###############################
####  OGGM configuration  #####
###############################
cfg.initialize() # initialize OGGM configuration

PATHS = PyDict(cfg."PATHS")  # OGGM PATHS
home_dir = cd(pwd, "../../../../..")
PATHS["working_dir"] = joinpath(home_dir, "Python/OGGM_data")  # Choose own custom path for the OGGM data
PARAMS = PyDict(cfg."PARAMS")

# Multiprocessing 
PARAMS["prcp_scaling_factor"], PARAMS["ice_density"], PARAMS["continue_on_error"]
PARAMS["use_multiprocessing"] = true # Let's use multiprocessing for OGGM

###############################################################
###########################  MAIN #############################
###############################################################

# Defining glaciers to be modelled with RGI IDs
# RGI60-11.03638 # Argentière glacier
# RGI60-11.01450 # Aletsch glacier
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]

### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
# Where to fetch the pre-processed directories

# TODO: make this constant
base_url = ("https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/elev_bands")

# Where to fetch the pre-processed directories

# TODO: change to Lilian's version in notebook (ODINN_MB.ipynb)
# use elevation band  flowlines
# gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=2,
#                                           prepro_border=10,
#                                           prepro_base_url=base_url,
#                                           prepro_rgi_version="62")

(@isdefined gdirs) || (gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_border=40)) 
# gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_border=40)

gdir = gdirs[2]
println("Path to the DEM:", gdir.get_filepath("dem"))

# Obtain ice thickness inversion
if !@isdefined glacier_gd
    list_talks = [
        # tasks.glacier_masks,
        # tasks.compute_centerlines,
        # tasks.initialize_flowlines,
        # tasks.compute_downstream_line,
        tasks.gridded_attributes,
        tasks.gridded_mb_attributes,
        # tasks.prepare_for_inversion,  # This is a preprocessing task
        # tasks.mass_conservation_inversion,  # This gdirsdoes the actual job
        # tasks.filter_inversion_output,  # This smoothes the thicknesses at the tongue a little
        # tasks.distribute_thickness_per_altitude,
        bedtopo.add_consensus_thickness   # Use consensus ice thicknesses from Farinotti et al. (2019)
    ]
    for task in list_talks
        # The order matters!
        workflow.execute_entity_task(task, gdirs)
    end
end

# Plot glacier domain<
# graphics.plot_domain(gdirs)
# graphics.plot_distributed_thickness(gdir)     

(@isdefined glacier_gd) || (glacier_gd = xr.open_dataset(gdir.get_filepath("gridded_data")))

# Broadcast necessary variables to all workers
sendto(workers(), glacier_gd=glacier_gd)
sendto(workers(), gdir=gdir)

@everywhere begin
nx = glacier_gd.y.size # glacier extent
ny = glacier_gd.x.size # really weird, but this is inversed 
const Δx = gdir.grid.dx
const Δy = gdir.grid.dy
end # @everywhere

MBsandbox = pyimport("MBsandbox.mbmod_daily_oneflowline")

### Generate fake annual long-term temperature time series  ###
# This represents the long-term average air temperature, which will be used to 
# drive changes in the `A` value of the SIA
(@isdefined temp_series) || (const temp_series, norm_temp_series = fake_temp_series(t₁))
# A_series = []
# for temps in temp_series
#     push!(A_series, A_fake.(temps))
# end
# display(Plots.plot(temp_series, xaxis="Years", yaxis="Long-term average air temperature", title="Fake air temperature time series"))
# display(Plots.plot(A_series, xaxis="Years", yaxis="A", title="Fake A reference time series"))

# Determine initial conditions
(@isdefined H₀) || (const H₀ = glacier_gd.consensus_ice_thickness.data) # initial ice thickness conditions for forward model
fillNaN!(H₀) # Fill NaNs with 0s to have real boundary conditions
(@isdefined B) || (const B = glacier_gd.topo.data .- H₀) # bedrock

# Run forward model for selected glaciers
if create_ref_dataset 
    println("Generating reference dataset for training...")
 
    # Compute reference dataset in parallel
    @everywhere solver = Ralston()
    H_refs, V̄x_refs, V̄y_refs = generate_ref_dataset(temp_series, H₀)
        
    println("Saving reference data")
    jldsave(joinpath(root_dir, "data/PDE_refs_ODINN.jld2"); H_refs, V̄x_refs, V̄y_refs)
end

# Load stored PDE reference datasets
PDE_refs = load(joinpath(root_dir, "data/PDE_refs_ODINN.jld2"))


