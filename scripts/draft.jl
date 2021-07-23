cd(@__DIR__)
using Pkg; Pkg.activate("../."); Pkg.instantiate()
using Plots; gr()
using SparseArrays
using Statistics
using ModelingToolkit
using LinearAlgebra
using HDF5
using JLD
using Infiltrator

### Global parameters  ###
include("helpers/parameters.jl")
### Types  ###
include("helpers/types.jl")
### Iceflow forward model  ###
# (includes utils.jl as well)
include("helpers/iceflow.jl")

# Load the HDF5 file with Harry's simulated data
root_dir = pwd()
argentiere_f = h5open(joinpath(root_dir, "../data/Argentiere_2003-2100_aflow2e-16_50mres_rcp2.6.h5"), "r")

# Fill the Glacier structure with the retrieved data
argentiere = Glacier(HDF5.read(argentiere_f["bed"])[begin:end-2,:],
                     HDF5.read(argentiere_f["thick_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["vel_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["s_apply_hist"])[begin:end-2,:,2:end],
                     0, 0)

# Update mass balance data with NaNs
MB_plot = copy(argentiere.MB)
voidfill!(MB_plot, argentiere.MB[1,1,1])

# Domain size
nx = size(argentiere.bed)[1]
ny = size(argentiere.bed)[2]


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

p = (Δx, Δy, Γ, A, B, v, argentiere.MB, MB_avg, C, α) 
H = copy(H₀)

sigmoid_A(x) = minA + (maxA - minA) / ( 1 + exp(-x) )
hyparams, UA = create_NNs()

all_MB = LinRange(0,3,10)
all_UA = UA(all_MB')[1,:]
plot(all_MB, all_UA, xlim=(0,2), ylim=(minA, maxA),markershape=:circle,markersize=4)

sigmoid_A.(all_MB)