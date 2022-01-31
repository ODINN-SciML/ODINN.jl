#= Glacier ice dynamics toy model

Test with ideal data of a hybrid glacier ice dynamics model based on neural networks
that respect the Shallow Ice Approximation, mixed with model interpretation using 
SINDy (Brunton et al., 2016).

=#

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
    Pkg.activate("../.");
    Pkg.instantiate()
end

@everywhere begin 
using Statistics
using LinearAlgebra
using HDF5  
using JLD
using Random 
using OrdinaryDiffEq
using DiffEqFlux
using Flux
using Distributed
using Tullio
using RecursiveArrayTools
using Infiltrator
using Plots

# using Profile
# using Logging: global_logger
# using TerminalLoggers: TerminalLogger
# global_logger(TerminalLogger())
# using ProgressLogging

### Types  ###
include("helpers/types.jl")
end # @everywhere 

### Global parameters  ###
include("helpers/parameters.jl")

### Iceflow forward model  ###
# (includes utils.jl as well)
include("helpers/iceflow_DiffEqs.jl")
### Climate data processing  ###
include("helpers/climate.jl")

###############################################################
###########################  MAIN #############################
###############################################################

@everywhere begin
    
# Load the HDF5 file with Harry's simulated data
cd(@__DIR__)
root_dir = cd(pwd, "..")
argentiere_f = h5open(joinpath(root_dir, "data/Argentiere_2003-2100_aflow2e-16_50mres_rcp2.6.h5"), "r")

# Fill the Glacier structure with the retrieved data
argentiere = Glacier(HDF5.read(argentiere_f["bed"])[begin:end-2,:],
                     HDF5.read(argentiere_f["thick_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["vel_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["s_apply_hist"])[begin:end-2,:,2:end],
                     0, 0)
    
# Domain size
const nx = size(argentiere.bed)[1]
const ny = size(argentiere.bed)[2]

end # @everywhere

# Update mass balance data with NaNs
#MB_plot = copy(argentiere.MB)
#voidfill!(MB_plot, argentiere.MB[1,1,1])
# Interpolate mass balance to daily values
#MB_weekly = interpolate(argentiere.MB/54, (NoInterp(), NoInterp(), BSpline(Linear())))

# Get the annual ELAs based on the mass balance data
#ELAs = get_annual_ELAs(argentiere.MB, argentiere.bed .+ argentiere.thick)


###  Plot initial data  ###
# Argentière bedrock
#hm01 = heatmap(argentiere.bed, c = :turku, title="Bedrock")
# Argentière ice thickness for an individual year
#hm02 = heatmap(argentiere.thick[:,:,1], c = :ice, title="Ice thickness")
# Surface velocities
#hm03 = heatmap(argentiere.vel[:,:,15], c =:speed, title="Ice velocities")
#hm04 = heatmap(MB_plot[:,:,90], c = cgrad(:balance,rev=true), clim=(-12,12), title="Mass balance")
#hm0 = plot(hm01,hm02,hm03,hm04, layout=4, aspect_ratio=:equal, xlims=(0,180))
#display(hm0)

### Generate fake annual long-term temperature time series  ###
# This represents the long-term average air temperature, which will be used to 
# drive changes in the `A` value of the SIA
const temp_series, norm_temp_series =  fake_temp_series(t₁) 

# pts = Plots.plot(temp_series, xaxis="Years", yaxis="Long-term average air temperature", title="Fake air temperature time series")
# pas = Plots.plot(A_series, xaxis="Years", yaxis="A", title="Fake A reference time series")


#### Choose the example to run  #####
example = "Argentiere"
# example = "Gaussian" # Fake

if example == "Argentiere"

    @everywhere begin
    const B  = copy(argentiere.bed)
    const H₀ = copy(argentiere.thick[:,:,1])
 
    # Spatial and temporal differentials
    const Δx, Δy = 50, 50 #m (Δx = Δy)
    end # @everywhere

    #=
    MB_avg = []
    for year in 1:length(argentiere.MB[1,1,:])
        MB_buff = buffer_mean(argentiere.MB, year)
        voidfill!(MB_buff, MB_buff[1,1], 0)
        push!(MB_avg, MB_buff)
    end 

    # Fill areas outside the glacier with NaN values for scalar training
    voidfill!(MB_avg, argentiere.MB[1,1,1])
    =#
    
elseif example == "Gaussian"
    
    const B = zeros(Float64, (nx, ny))
    const σ = 1000
    H₀ = [ 250 * exp( - ( (i - nx/2)^2 + (j - ny/2)^2 ) / σ ) for i in 1:nx, j in 1:ny ]    
    
    # Spatial and temporal differentials
    Δx = Δy = 50 #m (Δx = Δy)    

end


# We generate the reference dataset using fake know laws
if create_ref_dataset 
    println("Generating reference dataset for training...")
  
    # Compute reference dataset in parallel
    const H_refs = generate_ref_dataset(temp_series, H₀)
    
    # Save reference plots
    for (temps, H) in zip(temp_series, H_refs)
        ### Glacier ice thickness evolution  ### Not that useful
        # hm11 = heatmap(H₀, c = :ice, title="Ice thickness (t=0)")
        # hm12 = heatmap(H, c = :ice, title="Ice thickness (t=$t₁)")
        # hm1 = Plots.plot(hm11,hm12, layout=2, aspect_ratio=:equal, size=(800,350),
        #     colorbar_title="Ice thickness (m)",
        #     clims=(0,maximum(H₀)), link=:all)
        # display(hm1)
        
        ###  Glacier ice thickness difference  ###
        lim = maximum( abs.(H[end] .- H₀) )
        hm = heatmap(H[end] .- H₀, c = cgrad(:balance,rev=true), aspect_ratio=:equal,
            clim = (-lim, lim),
            title="Variation in ice thickness")

        tempn = floor(mean(temps))
        println("Saving reference_$tempn.png...")
        savefig(hm,joinpath(root_dir,"plots/references","reference_$tempn.png"))
    end
        
    println("Saving reference data")
    save(joinpath(root_dir, "data/H_refs.jld"), "H_refs", H_refs)

else 
   const H_refs = load(joinpath(root_dir, "data/H_refs.jld"))["H_refs"]
end


# We train an UDE in order to learn and infer the fake laws
if train_UDE
    
    println("Running forward UDE ice flow model...\n")
    
    # temp_values = [mean(temps) for temps in temp_series]'
    # norm_temp_values = [mean(temps) for temps in norm_temp_series]'
    # plot(temp_values', A_fake.(temp_values)', label="Fake A")

    # Define NN for the UDE
    UA = create_NNs()
    θ = initial_params(UA)

    # old_trained = predict_A̅(UA, θ, norm_temp_values)' #A_fake.(temp_values)'
    current_epoch = 1

    # Train iceflow UDE with in parallel
    iceflow_trained = train_iceflow_UDE(H₀, UA, θ, H_refs, temp_series)
    θ_trained = iceflow_trained.minimizer


    ###################################################################
    ########################  PLOTS    ################################
    ###################################################################

    pred_A = predict_A̅(UA, θ_trained, collect(-20.0:0.0)')
    pred_A = [pred_A...] # flatten
    true_A = A_fake(-20:0.0)

    plot(true_A, label="True A")
    train_final = plot!(pred_A, label="Predicted A")
    savefig(train_final,joinpath(root_plots,"training","final_model.png"))

end