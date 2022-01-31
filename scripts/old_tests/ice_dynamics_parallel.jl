#= Glacier ice dynamics toy model

Test with ideal data of a hybrid glacier ice dynamics model based on neural networks
that respect the Shallow Ice Approximation, mixed with model interpretation using 
SINDy (Brunton et al., 2016).

=#

## Environment and packages
using Distributed
processes = 4

if nprocs() < processes
    addprocs(processes - nprocs(); exeflags="--project")
end

println("Number of cores: ", nprocs())
println("Number of workers: ", nworkers())

@everywhere begin 
    #cd(@__DIR__)
    using Pkg 
    Pkg.activate("../.");
    Pkg.instantiate()
end
    
@everywhere begin 
    using Plots; gr()
    ENV["GKSwstype"] = "nul"
    using Statistics
    using LinearAlgebra
    using HDF5
    using JLD
    using Infiltrator
    using PyCall # just for compatibility with utils.jl
    using Random 
    
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
argentiere_f = h5open(joinpath(root_dir, "data/Argentiere_2003-2100_aflow2e-16_50mres_rcp2.6.h5"), "r")

# Fill the Glacier structure with the retrieved data
argentiere = Glacier(HDF5.read(argentiere_f["bed"])[begin:end-2,:],
                     HDF5.read(argentiere_f["thick_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["vel_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["s_apply_hist"])[begin:end-2,:,2:end],
                     0, 0)
    
end # @everywhere

# Update mass balance data with NaNs
MB_plot = copy(argentiere.MB)
voidfill!(MB_plot, argentiere.MB[1,1,1])
# Interpolate mass balance to daily values
#MB_weekly = interpolate(argentiere.MB/54, (NoInterp(), NoInterp(), BSpline(Linear())))

# Get the annual ELAs based on the mass balance data
#ELAs = get_annual_ELAs(argentiere.MB, argentiere.bed .+ argentiere.thick)

# Domain size
@everywhere nx = size(argentiere.bed)[1]
@everywhere ny = size(argentiere.bed)[2];


###  Plot initial data  ###
# Argentière bedrock
hm01 = heatmap(argentiere.bed, c = :turku, title="Bedrock")
# Argentière ice thickness for an individual year
hm02 = heatmap(argentiere.thick[:,:,1], c = :ice, title="Ice thickness")
# Surface velocities
hm03 = heatmap(argentiere.vel[:,:,15], c =:speed, title="Ice velocities")
hm04 = heatmap(MB_plot[:,:,90], c = cgrad(:balance,rev=true), clim=(-12,12), title="Mass balance")
hm0 = plot(hm01,hm02,hm03,hm04, layout=4, aspect_ratio=:equal, xlims=(0,180))
#display(hm0)

### Generate fake annual long-term temperature time series  ###
# This represents the long-term average air temperature, which will be used to 
# drive changes in the `A` value of the SIA
@everywhere temp_series, norm_temp_series =  fake_temp_series(t₁)

A_series = []
for temps in temp_series
    push!(A_series, A_fake.(temps))
end

pts = Plots.plot(temp_series, xaxis="Years", yaxis="Long-term average air temperature", title="Fake air temperature time series")
pas = Plots.plot(A_series, xaxis="Years", yaxis="A", title="Fake A reference time series")


#### Choose the example to run  #####
example = "Argentiere"
# example = "Gaussian" # Fake

if example == "Argentiere"

    @everywhere B  = copy(argentiere.bed)
    @everywhere H₀ = copy(argentiere.thick[:,:,1])
 
    # Spatial and temporal differentials
    @everywhere Δx = Δy = 50 #m (Δx = Δy)

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
    
    # Spatial and temporal differentials
    Δx = Δy = 50 #m (Δx = Δy)    

end

### We perform the simulations with an explicit forward mo  ###

@everywhere begin
    ts = collect(1:t₁)
    
    #gref = (H_ref, V_ref, ts)
    gref = Dict("H"=>[], "V"=>[], "timestamps"=>ts)
end

# We generate the reference dataset using fake know laws
if create_ref_dataset 
    println("Generating reference dataset for training...")
  
    # Compute reference dataset in parallel
    @time generate_ref_dataset(temp_series, gref, H₀, t)
    
    # @time @everywhere glacier_refs = ref_dataset(temp_series, gref, H₀, p, t, t₁, ref_n)
    
    println("Saving reference data")
    save(joinpath(root_dir, "data/glacier_refs.jld"), "glacier_refs", glacier_refs)

else 
    glacier_refs = load(joinpath(root_dir, "data/glacier_refs.jld"))["glacier_refs"]
end



# We define the training itenerary
#temps_list = []
#for i in 1:hyparams.epochs
#    temps = LinRange{Int}(1, length(temp_series), length(temp_series))[Random.shuffle(1:end)]
#    temps_list = vcat(temps_list, temps)
#end

# We train an UDE in order to learn and infer the fake laws
if train_UDE
    
    println("Running forward UDE ice flow model...\n")
    
    let
    temp_values = [mean(temps) for temps in temp_series]'
    norm_temp_values = [mean(temps) for temps in norm_temp_series]'
    plot(temp_values', A_fake.(temp_values)', label="Fake A")
    hyparams, UA = create_NNs()
    old_trained = predict_A̅(UA, norm_temp_values)' #A_fake.(temp_values)'
    
    #trackers = Dict("losses"=>[], "losses_batch"=>[],
    #                "current_batch"=>1, "grad_batch"=>[])

    # Diagnosis plot after each full epochs
    #display(scatter!(temp_values', predict_A̅(UA, temp_values)', yaxis="A", xaxis="Year", label="Trained NN"))#, ylims=(3e-17,8e-16)))

    # Train iceflow UDE
    for i in 1:hyparams.epochs
        println("\nEpoch #", i, "\n")
        
        # Randomize order of glaciers in the batch
        idxs = Random.shuffle(1:length(temp_series))
        # idxs = 1:length(temp_series)
        #temps = LinRange{Int}(1, length(temp_series), length(temp_series))[Random.shuffle(1:end)]
        
        # Train UDE batch in parallel
        @time train_batch_iceflow_UDE(H₀, UA, glacier_refs, temp_series, hyparams, idxs)  
        
        # Update NN weights after batch completion 
        @time update_UDE_batch!(UA, loss_UAs, back_UAs)
        
        # Plot evolution of training
        plot_training!(old_trained, UA, loss_UAs, temp_values, norm_temp_values)

    end
end
end # let


###################################################################
########################  PLOTS    ################################
###################################################################
