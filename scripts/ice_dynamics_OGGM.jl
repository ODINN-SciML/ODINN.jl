#= Glacier ice dynamics toy model

Test with ideal data of a hybrid glacier ice dynamics model based on neural networks
that respect the Shallow Ice Approximation, mixed with model interpretation using 
SINDy (Brunton et al., 2016).

=#

###############################################
############  JULIA ENVIRONMENT  ##############
###############################################
cd(@__DIR__)
using Pkg; Pkg.activate("../."); 
Pkg.instantiate()
using Plots; gr()
using SparseArrays
using Statistics
using LinearAlgebra
using HDF5
using JLD
using Infiltrator
using Dates # to provide correct Julian time slices 

################################################
############  PYTHON ENVIRONMENT  ##############
################################################

## Set up Python environment
# Choose own Python environment with OGGM's installation
# Use same path as "which python" in shell
ENV["PYTHON"] = "/Users/Bolib001/miniconda3/envs/oggm_env/bin/python3.9" 
Pkg.build("PyCall") 
using PyCall
using PyPlot # needed for Matplotlib plots

# Essential Python libraries
np = pyimport("numpy")
xr = pyimport("xarray")
matplotlib = pyimport("matplotlib")
matplotlib.use("Qt5Agg")
# plt = pyimport("matplotlib.pyplot")

# Import OGGM sub-libraries in Julia
cfg = pyimport("oggm.cfg")
utils = pyimport("oggm.utils")
workflow = pyimport("oggm.workflow")
tasks = pyimport("oggm.tasks")
graphics = pyimport("oggm.graphics")
MBsandbox = pyimport("MBsandbox.mbmod_daily_oneflowline")

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
PATHS["working_dir"] = "/Users/Bolib001/Jordi/Python/OGGM_data"  # Choose own custom path for the OGGM data

###############################################################
###########################  MAIN #############################
###############################################################

# Defining glaciers to be modelled with RGI IDs
# RGI60-11.03638 # Argentière glacier
# RGI60-11.01450 # Aletsch glacier
rgi_ids = ["RGI60-11.03638"]

# Where to fetch the pre-processed directories
gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_border=80)

gdir = gdirs[1]

# Obtain ice thickness inversion
list_talks = [
    # tasks.glacier_masks,
    # tasks.compute_centerlines,
    # tasks.initialize_flowlines,
    # tasks.compute_downstream_line,
    tasks.prepare_for_inversion,  # This is a preprocessing task
    tasks.mass_conservation_inversion,  # This does the actual job
    tasks.filter_inversion_output,  # This smoothes the thicknesses at the tongue a little
    tasks.distribute_thickness_per_altitude
]
for task in list_talks
    # The order matters!
    workflow.execute_entity_task(task, gdirs)
end
# Plot glacier domain
graphics.plot_domain(gdirs)
graphics.plot_distributed_thickness(gdir)         


### We perform the simulations with an explicit forward mo  ###
# Gather simulation parameters
p = (Δx, Δy, Γ, A, B, v, argentiere.MB, MB_avg, C, α, var_format) 
H = copy(H₀)

# We generate the reference dataset using fake know laws
if create_ref_dataset 
    ts = collect(1:t₁)
    H_ref = Dict("H"=>[], "timestamps"=>ts)
    @time H = iceflow!(H,H_ref,p,t,t₁)
else 
    H_ref = load(joinpath(root_dir, "data/H_ref.jld"))["H"]
end

# We train an UDE in order to learn and infer the fake laws
if train_UDE
    hyparams, UA = create_NNs()


    period = length(MB_avg)
    ŶA = []
    MB_nan = deepcopy(MB_avg)
    for i in 1:period
        MB_nan[i][MB_nan[i] .== 0] .= NaN
        append!(ŶA, A_fake(MB_nan[i], size(H), "scalar"))
    end
    plot(ŶA, yaxis="A", xaxis="Year", label="fake A")
    # plot(fakeA, 0, t₁, label="fake")
    initial_NN = []
    for i in 1:period
        append!(initial_NN, predict_A(UA, MB_nan, i, "scalar"))
    end
    display(plot!(initial_NN, label="initial NN"))


    # Train iceflow UDE
    iceflow_UDE!(H,H_ref,UA,hyparams,p,t,t₁)

end



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


