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
using Statistics
using LinearAlgebra
using Random 
using HDF5  
using JLD
using OrdinaryDiffEq
using DiffEqFlux
using Flux
using Distributed
using Tullio
using RecursiveArrayTools
using Infiltrator
using Plots

const t₁ = 5                 # number of simulation years 
const ρ = 900                     # Ice density [kg / m^3]
const g = 9.81                    # Gravitational acceleration [m / s^2]
const n = 3                      # Glen's flow law exponent
const maxA = 8e-16
const minA = 3e-17
const maxT = 1
const minT = -25

create_ref_dataset = false

# Include all functions
include("iceflow.jl")

##################################################
#### Generate reference dataset ####
##################################################
# Load the HDF5 file with Harry's simulated data
cd(@__DIR__)
root_dir = cd(pwd, "../..")
argentiere_f = h5open(joinpath(root_dir, "data/Argentiere_2003-2100_aflow2e-16_50mres_rcp2.6.h5"), "r")

struct Glacier
    bed::Array{Float32}    # bedrock height
    thick::Array{Float32}  # ice thickness
    vel::Array{Float32}    # surface velocities
    MB::Array{Float32}     # surface mass balance
    lat::Float32
    lon::Float32
end

# Fill the Glacier structure with the retrieved data
argentiere = Glacier(HDF5.read(argentiere_f["bed"])[begin:end-2,:],
                     HDF5.read(argentiere_f["thick_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["vel_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["s_apply_hist"])[begin:end-2,:,2:end],
                     0, 0)
    
# Domain size
const nx = size(argentiere.bed)[1]
const ny = size(argentiere.bed)[2]

const B  = copy(argentiere.bed)
const H₀ = copy(argentiere.thick[:,:,1])

# Spatial and temporal differentials
const Δx, Δy = 50, 50 #m (Δx = Δy)

# ensemble = EnsembleSerial()
ensemble = EnsembleDistributed()
const minA_out = 0.3
const maxA_out = 8
sigmoid_A(x) = minA_out + (maxA_out - minA_out) / ( 1 + exp(-x) )
end # @everywhere

const temp_series, norm_temp_series = fake_temp_series(t₁)

if create_ref_dataset 
    println("Generating reference dataset for training...")
  
    # Compute reference dataset in parallel
    H_refs = generate_ref_dataset(temp_series, H₀)
    
    # Save reference plots
    # for (temps, H) in zip(temp_series, H_refs)
        
    #     ###  Glacier ice thickness difference  ###
    #     lim = maximum( abs.(H[end] .- H₀) )
    #     hm = heatmap(H[end] .- H₀, c = cgrad(:balance,rev=true), aspect_ratio=:equal,
    #         clim = (-lim, lim),
    #         title="Variation in ice thickness")

    #     tempn = floor(mean(temps))
    #     println("Saving reference_$tempn.png...")
    #     savefig(hm,joinpath(root_dir,"plots/references","reference_$tempn.png"))
    # end
        
    println("Saving reference data")
    save(joinpath(root_dir, "data/H_refs.jld"), "H_refs", H_refs)

else
   H_refs = load(joinpath(root_dir, "data/H_refs.jld"))["H_refs"]
end
    
# Train UDE
UA = FastChain(
        FastDense(1,3, x->tanh.(x)),
        FastDense(3,10, x->tanh.(x)),
        FastDense(10,3, x->tanh.(x)),
        FastDense(3,1, sigmoid_A)
    )
    
θ = initial_params(UA)
const epochs = 30
current_epoch = 1
const η = 0.005
batch_size = length(temp_series)

cd(@__DIR__)
const root_plots = cd(pwd, "../../plots")
# Train iceflow UDE in parallel
@time iceflow_trained = train_iceflow_UDE(H₀, UA, θ, H_refs, temp_series)
θ_trained = iceflow_trained.minimizer

pred_A = predict_A̅(UA, θ_trained, collect(-20.0:0.0)')
pred_A = [pred_A...] # flatten
true_A = A_fake(-20:0.0)

plot(true_A, label="True A")
train_final = plot!(pred_A, label="Predicted A")
savefig(train_final,joinpath(root_plots,"training","final_model.png"))