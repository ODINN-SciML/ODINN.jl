#= Glacier ice dynamics toy model

Test with ideal data of a hybrid glacier ice dynamics model based on neural networks
that respect the Shallow Ice Approximation, mixed with model interpretation using 
SINDy (Brunton et al., 2016).

=#

###############################################
###########  ACTIVATE ENVIRONMENT  ############
###############################################

cd(@__DIR__)
using Pkg 
Pkg.activate("../../.");
Pkg.instantiate()

################################################
############  PYTHON ENVIRONMENT  ##############
################################################

## Set up Python environment
# Choose own Python environment with OGGM's installation
# Use same path as "which python" in shell
# ENV["PYTHON"] = "/Users/Bolib001/miniconda3/envs/oggm_env/bin/python3.9" 
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
const processes = 10

if nprocs() < processes
    addprocs(processes - nprocs(); exeflags="--project")
end

println("Number of cores: ", nprocs())
println("Number of workers: ", nworkers())

@everywhere begin 
    cd(@__DIR__)
    using Pkg 
    Pkg.activate("../../.");
    Pkg.instantiate()
end

@everywhere begin 
using Plots; gr()
using OrdinaryDiffEq
using Tullio
using RecursiveArrayTools
using Statistics
using LinearAlgebra
using HDF5
using JLD
using Infiltrator
using Dates # to provide correct Julian time slices 

###############################################
#############    PARAMETERS     ###############
###############################################

const t₁ = 5                 # number of simulation years 
const ρ = 900                     # Ice density [kg / m^3]
const g = 9.81                    # Gravitational acceleration [m / s^2]
const n = 3                      # Glen's flow law exponent
const maxA = 8e-16
const minA = 3e-17
const maxT = 1
const minT = -25

###############################################
#############  ODINN LIBRARIES  ###############
###############################################

### Iceflow forward model  ###
# (includes utils.jl as well)
include("iceflow.jl")

cd(@__DIR__)
root_dir = cd(pwd, "../..")

### Climate data processing  ###
include(joinpath(root_dir, "scripts/helpers/climate.jl"))
end # @everywhere

cfg.initialize() # initialize OGGM configuration

PATHS = PyDict(cfg."PATHS")  # OGGM PATHS
home_dir = cd(pwd, "../../../..")
PATHS["working_dir"] = joinpath(home_dir, "/Python/OGGM_data")  # Choose own custom path for the OGGM data
PARAMS = PyDict(cfg."PARAMS")

# Multiprocessing 
PARAMS["use_multiprocessing"] = false # Let's use multiprocessing for OGGM
# ensemble = EnsembleSerial()
ensemble = EnsembleDistributed() # multiprocessing for ODINN

###############################################################
###########################  MAIN #############################
###############################################################

# Defining glaciers to be modelled with RGI IDs
# RGI60-11.03638 # Argentière glacier
# RGI60-11.01450 # Aletsch glacier
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]

### Initialize glacier directory to obtain DEM and ice thickness inversion  ###
# Where to fetch the pre-processed directories
gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_border=10)

gdir = gdirs[1]

# Obtain ice thickness inversion
list_talks = [
    # tasks.glacier_masks,
    # tasks.compute_centerlines,
    # tasks.initialize_flowlines,
    # tasks.compute_downstream_line,
    tasks.prepare_for_inversion,  # This is a preprocessing task
    tasks.mass_conservation_inversion,  # This does the actual job
    # tasks.filter_inversion_output,  # This smoothes the thicknesses at the tongue a little
    tasks.distribute_thickness_per_altitude
]
for task in list_talks
    # The order matters!
    workflow.execute_entity_task(task, gdirs)
end
# Plot glacier domain
graphics.plot_domain(gdirs)
graphics.plot_distributed_thickness(gdir)     

glacier_gd = xr.open_dataset(gdir.get_filepath("gridded_data"))
nx = glacier_gd.x.size # glacier extent
ny = glacier_gd.y.size

### Generate fake annual long-term temperature time series  ###
# This represents the long-term average air temperature, which will be used to 
# drive changes in the `A` value of the SIA
temp_series =  fake_temp_series(t₁)
A_series = []
for temps in temp_series
    push!(A_series, A_fake.(temps))
end
display(Plots.plot(temp_series, xaxis="Years", yaxis="Long-term average air temperature", title="Fake air temperature time series"))
display(Plots.plot(A_series, xaxis="Years", yaxis="A", title="Fake A reference time series"))

# Determine initial conditions
H = glacier_gd.distributed_thickness.data # initial ice thickness conditions for forward model
B = glacier_gd.topo.data - glacier_gd.distributed_thickness.data # bedrock

# Run forward model for selected glaciers


