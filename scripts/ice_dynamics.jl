#= Glacier ice dynamics toy model

Test with ideal data of a hybrid glacier ice dynamics model based on neural networks
that respect the Shallow Ice Approximation, mixed with model interpretation using 
SINDy (Brunton et al., 2016).

=#

## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("../."); 
# Pkg.instantiate()
using Plots; gr()
using SparseArrays
using Statistics
using LinearAlgebra
using HDF5
using JLD
using Infiltrator
#using Test
# using Flux
# using Flux: @epochs
#using BenchmarkTools
#using PaddedViews
#using Random
#using Debugger
# using Flux, DiffEqFlux, DataDrivenDiffEq

# using Zygote
#using DifferentialEquations
#using ComponentArrays
#using Parameters: @unpack
#using Interpolations
#using CUDA
#using Measures

### Global parameters  ###
include("helpers/parameters.jl")
### Types  ###
include("helpers/types.jl")
### Iceflow forward model  ###
# (includes utils.jl as well)
include("helpers/iceflow.jl")

###############################################################
###########################  MAIN #############################
###############################################################

# Load the HDF5 file with Harry's simulated data
root_dir = cd(pwd, "..")
argentiere_f = h5open(joinpath(root_dir, "data/Argentiere_2003-2100_aflow2e-16_50mres_rcp2.6.h5"), "r")

# Fill the Glacier structure with the retrieved data
argentiere = Glacier(HDF5.read(argentiere_f["bed"])[begin:end-2,:],
                     HDF5.read(argentiere_f["thick_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["vel_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["s_apply_hist"])[begin:end-2,:,2:end],
                     0, 0)

# Update mass balance data with NaNs
MB_plot = copy(argentiere.MB)
voidfill!(MB_plot, argentiere.MB[1,1,1])
# Interpolate mass balance to daily values
#MB_weekly = interpolate(argentiere.MB/54, (NoInterp(), NoInterp(), BSpline(Linear())))

# Get the annual ELAs based on the mass balance data
#ELAs = get_annual_ELAs(argentiere.MB, argentiere.bed .+ argentiere.thick)

# Domain size
nx = size(argentiere.bed)[1]
ny = size(argentiere.bed)[2];


###  Plot initial data  ###
# Argentière bedrock
hm01 = heatmap(argentiere.bed, c = :turku, title="Bedrock")
# Argentière ice thickness for an individual year
hm02 = heatmap(argentiere.thick[:,:,1], c = :ice, title="Ice thickness")
# Surface velocities
hm03 = heatmap(argentiere.vel[:,:,15], c =:speed, title="Ice velocities")
hm04 = heatmap(MB_plot[:,:,90], c = cgrad(:balance,rev=true), clim=(-12,12), title="Mass balance")
hm0 = plot(hm01,hm02,hm03,hm04, layout=4, aspect_ratio=:equal, xlims=(0,180))
#display(hm0)

#### Choose the example to run  #####
example = "Argentiere"
#example = "Gaussian" # Fake

if example == "Argentiere"

    B  = copy(argentiere.bed)
    H₀ = copy(argentiere.thick[:,:,1])
    v = zeros(size(argentiere.thick)) # surface velocities
 
    # Spatial and temporal differentials
    Δx = Δy = 50 #m (Δx = Δy)

    MB_avg = []
    for year in 1:length(argentiere.MB[1,1,:])
        MB_buff = buffer_mean(argentiere.MB, year)
        voidfill!(MB_buff, MB_buff[1,1], 0)
        push!(MB_avg, MB_buff)
    end 
    
elseif example == "Gaussian"
    
    B = zeros(Float64, (nx, ny))
    σ = 1000
    H₀ = [ 250 * exp( - ( (i - nx/2)^2 + (j - ny/2)^2 ) / σ ) for i in 1:nx, j in 1:ny ]    
    
    # Spatial and temporal differentials
    Δx = Δy = 50 #m (Δx = Δy)    

end

### We perform the simulations with an explicit forward mo  ###
# Gather simulation parameters
p = (Δx, Δy, Γ, A, B, v, argentiere.MB, MB_avg, C, α) 
H = copy(H₀)

# We generate the reference dataset using fake know laws
if create_ref_dataset 
    H_ref = Dict("H"=>[], "timestamps"=>[1,2,3])
    @time H = iceflow!(H,H_ref,p,t,t₁)
else 
    H_ref = load(joinpath(root_dir, "data/H_ref.jld"))["H"]
end

# We train an UDE in order to learn and infer the fake laws
if train_UDE
    hyparams, UA = create_NNs()
    iceflow_UDE!(H,H_ref,UA,hyparams,p,t,t₁)
end



###################################################################
########################  PLOTS    ################################
###################################################################

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


