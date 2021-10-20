#= Glacier ice dynamics toy model

Test with ideal data of a hybrid glacier ice dynamics model based on neural networks
that respect the Shallow Ice Approximation, mixed with model interpretation using 
SINDy (Brunton et al., 2016).

=#

###############################################
###########  ACTIVATE ENVIRONMENT  ############
###############################################
cd(@__DIR__)
root_dir = cd(pwd, "..")
using Pkg; Pkg.activate("../."); 
Pkg.instantiate()

################################################
############  PYTHON ENVIRONMENT  ##############
################################################

## Set up Python environment
# Choose own Python environment with OGGM's installation
# Use same path as "which python" in shell
# ENV["PYTHON"] = "/Users/Bolib001/miniconda3/envs/oggm_env/bin/python3.9" 
ENV["PYTHON"] = "/nethome/bolib001/.conda/envs/oggm_env/bin/python3.6"
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
using Plots; gr()
using SparseArrays
using Statistics
using LinearAlgebra
# using HDF5
using JLD
using Infiltrator
using Dates # to provide correct Julian time slices 

###############################################
#############  ODINN LIBRARIES  ###############
###############################################

### Global parameters  ###
include("helpers/parameters.jl")
### Types  ###
include("helpers/types.jl")
### Iceflow forward model  ###
# (includes utils.jl as well)
include("helpers/iceflow.jl")
### Climate data processing  ###
include("helpers/climate.jl")

cfg.initialize() # initialize OGGM configuration

PATHS = PyDict(cfg."PATHS")  # OGGM PATHS
PATHS["working_dir"] = "/nethome/bolib001/Python/OGGM_data"  # Choose own custom path for the OGGM data

###############################################################
###########################  MAIN #############################
###############################################################

# Defining glaciers to be modelled with RGI IDs
# RGI60-11.03638 # Argentière glacier
# RGI60-11.01450 # Aletsch glacier
rgi_ids = ["RGI60-11.03638"]

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
display(plot(temp_series, xaxis="Years", yaxis="Long-term average air temperature", title="Fake air temperature time series"))
display(plot(A_series, xaxis="Years", yaxis="A", title="Fake A reference time series"))

let
H = glacier_gd.distributed_thickness.data # initial ice thickness conditions for forward model
B = glacier_gd.topo.data - glacier_gd.distributed_thickness.data # bedrock

# We generate the reference dataset using fake know laws
if create_ref_dataset 
    println("Generating reference dataset for training...")
    ts = collect(1:t₁)
    gref = Dict("H"=>[], "V"=>[], "timestamps"=>ts)
    glacier_refs = []

    for temps in temp_series
        println("Reference simulation with temp ≈ ", mean(temps))
        glacier_ref = copy(gref)
        # Gather simulation parameters
        p = (Δx, Δy, Γ, A, B, temps, C, α) 
        # Perform reference imulation with forward model 
        @time H, V = iceflow!(H,glacier_ref,p,t,t₁)
        push!(glacier_refs, glacier_ref)
    end

    println("Saving reference data")
    save(joinpath(root_dir, "data/glacier_refs.jld"), "glacier_refs", glacier_refs)
else 
    glacier_ref = load(joinpath(root_dir, "data/glacier_refs.jld"))
end

# We train an UDE in order to learn and infer the fake laws
if train_UDE
    println("Training UDEs...")
    hyparams, UA = create_NNs()

    # period = length(MB_avg)
    # ŶA = []
    # MB_nan = deepcopy(MB_avg)
    # for i in 1:period
    #     MB_nan[i][MB_nan[i] .== 0] .= NaN
    #     append!(ŶA, A_fake(MB_nan[i], size(H), "scalar"))
    # end
    # plot(ŶA, yaxis="A", xaxis="Year", label="fake A")
    # # plot(fakeA, 0, t₁, label="fake")
    # initial_NN = []
    # for i in 1:period
    #     append!(initial_NN, predict_A(UA, MB_nan, i, "scalar"))
    # end
    # display(plot!(initial_NN, label="initial NN"))

    global ts_i = 1
    for temps in temp_series
        println("UDE training with temp ≈ ", mean(temps))
        # Gather simulation parameters
        p = (Δx, Δy, Γ, A, B, temps, C, α) 
        # Train iceflow UDE
        iceflow_UDE!(H,glacier_ref,UA,hyparams,p,t,t₁)
        ts_i += 1
    end

end
end # let



###################################################################
########################  PLOTS    ################################
###################################################################

final_NN = []
for i in 1:period
    append!(final_NN, predict_A(UA, MB_nan, i, "scalar"))
end
plot(ŶA, yaxis="A", xaxis="Year", label="fake A")
display(plot!(final_NN, label="final NN"))

### Glacier ice thickness evolution  ###
hm11 = heatmap(H₀, c = :ice, title="Ice thickness (t=0)")
hm12 = heatmap(H, c = :ice, title="Ice thickness (t=$t₁)")
hm1 = plot(hm11,hm12, layout=2, aspect_ratio=:equal, size=(800,350),
      xlims=(0,180), ylims=(0,180), colorbar_title="Ice thickness (m)",
      clims=(0,maximum(H₀)), link=:all)
display(hm1)

###  Glacier ice thickness difference  ###
lim = maximum( abs.(H .- H₀) )
hm2 = heatmap(H .- H₀, c = cgrad(:balance,rev=true), aspect_ratio=:equal,
      xlims=(0,180), ylims=(0,180), clim = (-lim, lim),
      title="Variation in ice thickness")
display(hm2)


