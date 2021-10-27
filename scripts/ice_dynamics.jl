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
using PyCall # just for compatibility with utils.jl

### Global parameters  ###
include("helpers/parameters.jl")
### Types  ###
include("helpers/types.jl")
### Iceflow forward model  ###
# (includes utils.jl as well)
include("helpers/iceflow.jl")
### Climate data processing  ###
include("helpers/climate.jl")


###############################################################
###########################  MAIN #############################
###############################################################

# Load the HDF5 file with Harry's simulated data
root_dir = cd(pwd, "..")
res = 50 # m
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

#### Choose the example to run  #####
example = "Argentiere"
# example = "Gaussian" # Fake

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

    # Fill areas outside the glacier with NaN values for scalar training
    voidfill!(MB_avg, argentiere.MB[1,1,1])
    
elseif example == "Gaussian"
    
    B = zeros(Float64, (nx, ny))
    σ = 1000
    H₀ = [ 250 * exp( - ( (i - nx/2)^2 + (j - ny/2)^2 ) / σ ) for i in 1:nx, j in 1:ny ]    
    
    # Spatial and temporal differentials
    Δx = Δy = 50 #m (Δx = Δy)    

end

### We perform the simulations with an explicit forward mo  ###
# Gather simulation parameters
p = (Δx, Δy, Γ, A, B, v, argentiere.MB, MB_avg, C, α, var_format) 

ts = collect(1:t₁)
gref = Dict("H"=>[], "V"=>[], "timestamps"=>ts)
glacier_refs = []

# We generate the reference dataset using fake know laws
if create_ref_dataset 
    println("Generating reference dataset for training...")


    for temps in temp_series
        println("Reference simulation with temp ≈ ", mean(temps))
        glacier_ref = copy(gref)
        H = copy(H₀)
        # Gather simulation parameters
        p = (Δx, Δy, Γ, A, B, temps, C, α) 
        # Perform reference imulation with forward model 
        @time H, V̂ = iceflow!(H,glacier_ref,p,t,t₁)

        push!(glacier_refs, glacier_ref)

        ### Glacier ice thickness evolution  ###
        hm11 = heatmap(H₀, c = :ice, title="Ice thickness (t=0)")
        hm12 = heatmap(H, c = :ice, title="Ice thickness (t=$t₁)")
        hm1 = Plots.plot(hm11,hm12, layout=2, aspect_ratio=:equal, size=(800,350),
            colorbar_title="Ice thickness (m)",
            clims=(0,maximum(H₀)), link=:all)
        display(hm1)

        ###  Glacier ice thickness difference  ###
        lim = maximum( abs.(H .- H₀) )
        hm2 = heatmap(H .- H₀, c = cgrad(:balance,rev=true), aspect_ratio=:equal,
            clim = (-lim, lim),
            title="Variation in ice thickness")
        display(hm2)

    end

    println("Saving reference data")
    save(joinpath(root_dir, "data/glacier_refs.jld"), "glacier_refs", glacier_refs)

else 
    glacier_refs = load(joinpath(root_dir, "data/glacier_refs.jld"))["glacier_refs"]
end


# We train an UDE in order to learn and infer the fake laws
if train_UDE
    hyparams, UA = create_NNs()

    # Train iceflow UDE
    for (temps, glacier_ref) in zip(temp_series, glacier_refs)
        H = copy(H₀)
        # Gather simulation parameters
        p = (Δx, Δy, Γ, A, B, temps, C, α) 
        iceflow_UDE!(H,glacier_ref,UA,hyparams,p,t,t₁)
    end
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


