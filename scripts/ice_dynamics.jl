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
# using SparseArrays
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
temp_series, norm_temp_series =  fake_temp_series(t₁)

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

ts = collect(1:t₁)
gref = Dict("H"=>[], "V"=>[], "timestamps"=>ts)
glacier_refs = []

ts = collect(1:t₁)
gref = Dict("H"=>[], "V"=>[], "timestamps"=>ts)
glacier_refs = []

# We generate the reference dataset using fake know laws
if create_ref_dataset 
    println("Generating reference dataset for training...")

    for temps in temp_series
        println("Reference simulation with temp ≈ ", mean(temps))
        glacier_ref = deepcopy(gref)
        H = deepcopy(H₀)
        # Gather simulation parameters
        p = (Δx, Δy, Γ, A, B, temps, C, α) 
        # Perform reference imulation with forward model 
        @time H, V̂ = iceflow!(H,glacier_ref,p,t,t₁)

        push!(glacier_refs, glacier_ref)

        ### Glacier ice thickness evolution  ### Not that useful
        # hm11 = heatmap(H₀, c = :ice, title="Ice thickness (t=0)")
        # hm12 = heatmap(H, c = :ice, title="Ice thickness (t=$t₁)")
        # hm1 = Plots.plot(hm11,hm12, layout=2, aspect_ratio=:equal, size=(800,350),
        #     colorbar_title="Ice thickness (m)",
        #     clims=(0,maximum(H₀)), link=:all)
        # display(hm1)

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
    trackers = Dict("losses"=>[], "losses_batch"=>[],
                    "current_batch"=>1, "grad_batch"=>[])

    # Diagnosis plot after each full epochs
    #display(scatter!(temp_values', predict_A̅(UA, temp_values)', yaxis="A", xaxis="Year", label="Trained NN"))#, ylims=(3e-17,8e-16)))

    # Train iceflow UDE

    for i in 1:hyparams.epochs
        println("\nEpoch #", i, "\n")

        idxs = Random.shuffle(1:length(temp_series))
        # idxs = 1:length(temp_series)
        #temps = LinRange{Int}(1, length(temp_series), length(temp_series))[Random.shuffle(1:end)]

        for idx in idxs
            temps = temp_series[idx]
            norm_temps = norm_temp_series[idx]
            glacier_ref = glacier_refs[idx]
            println("\nTemperature in training: ", temps[1])

            # Gather simulation parameters
            p = (Δx, Δy, Γ, A, B, norm_temps, C, α) 
            iceflow_UDE!(H₀,glacier_ref,UA,hyparams,trackers,p,t,t₁)   
            
            predicted_A = predict_A̅(UA, [mean(norm_temps)]')[1]
            fake_A = A_fake(mean(temps)) 
            A_error = predicted_A - fake_A
            println("Predicted A: ", predicted_A)
            println("Fake A: ", fake_A)
            println("A error: ", A_error)

            if trackers["current_batch"] < hyparams.batchsize
                trackers["current_batch"] +=1 # increase batch
            else
                trackers["current_batch"] = 1

                 # Plot the evolution
                plot(temp_values', A_fake.(temp_values)', label="Fake A")
                # vline!([mean(temps)], label="Last temp")
                scatter!(temp_values', predict_A̅(UA, norm_temp_values)', yaxis="A", xaxis="Air temperature (°C)", label="Trained NN", color="red")#, ylims=(3e-17,8e-16)))
                pfunc = scatter!(temp_values', old_trained, label="Previous NN", color="grey", aspect=:equal, legend=:outertopright)#, ylims=(3e-17,8e-16)))
                ploss = plot(trackers["losses"], xlabel="Epoch", ylabel="Loss", aspect=:equal, legend=:outertopright, label="")
                ptrain = plot(pfunc, ploss, layout=(2,1))
                savefig(ptrain,joinpath(root_dir,"plots/training","epoch$i.png"))
                display(ptrain)
                old_trained = predict_A̅(UA, norm_temp_values)'

            end 
        end

    end
end
end # let


###################################################################
########################  PLOTS    ################################
###################################################################
