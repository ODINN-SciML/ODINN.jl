## Environment and packages
import Pkg
# activate project
Pkg.activate(dirname(Base.current_project()))
# make sure all packages are precompiled
Pkg.precompile()

# using Logging: global_logger
# using TerminalLoggers: TerminalLogger
# global_logger(TerminalLogger())

using Distributed
using ProgressMeter
const processes = 10

if nprocs() < processes
    addprocs(processes - nprocs(); exeflags="--project")
end

println("Number of cores: ", nprocs())
println("Number of workers: ", nworkers())

@everywhere begin 
    import Pkg
    Pkg.activate(dirname(Base.current_project()))
    Pkg.precompile()
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
using Tullio
using RecursiveArrayTools
using Infiltrator
using Plots
using ProgressMeter

const t₁ = 5                 # number of simulation years 
const ρ = 900                     # Ice density [kg / m^3]
const g = 9.81                    # Gravitational acceleration [m / s^2]
const n = 3                      # Glen's flow law exponent
const maxA = 8e-16
const minA = 3e-17
const maxT = 1
const minT = -25

create_ref_dataset = true
const noise = true # Add random noise to fake A law
rng_seed() = MersenneTwister(123) # random seed

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

(@isdefined B) || (const B  = copy(argentiere.bed))
(@isdefined H₀) || (const H₀ = copy(argentiere.thick[:,:,1]))

# Spatial and temporal differentials
const Δx, Δy = 50, 50 #m (Δx = Δy)

const minA_out = 0.3
const maxA_out = 8
sigmoid_A(x) = minA_out + (maxA_out - minA_out) / ( 1 + exp(-x) )
end # @everywhere

# Include all functions
include("iceflow.jl")

(@isdefined temp_series) || (const temp_series, norm_temp_series = fake_temp_series(t₁))

if create_ref_dataset 
    println("Generating reference dataset for training...")
  
    # Compute reference dataset in parallel
    @everywhere solver = Ralston()
    H_refs = generate_ref_dataset(temp_series, H₀)
    
    ## Save reference plots
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
        FastDense(1,3, x->softplus.(x)),
        FastDense(3,10, x->softplus.(x)),
        FastDense(10,3, x->softplus.(x)),
        FastDense(3,1, sigmoid_A)
    )
    
θ = initial_params(UA)
current_epoch = 1
η = 0.02
batch_size = length(temp_series)

# Settings for  distributed progress bars
# n_trajectories = batch_size
# progress = Progress(n_trajectories)
# const channel = RemoteChannel(()->Channel{Int}(n_trajectories))
# @everywhere const channel = $channel

cd(@__DIR__)
const root_plots = cd(pwd, "../../plots")
# Train iceflow UDE in parallel
# First train with ADAM to move the parameters into a favourable space
@everywhere solver = ROCK4()
train_settings = (RMSProp(η), 10) # optimizer, epochs
iceflow_trained = @time train_iceflow_UDE(H₀, UA, θ, train_settings, H_refs, temp_series)
θ_trained = iceflow_trained.minimizer

# Continue training with BFGS
train_settings = (BFGS(initial_stepnorm=0.01f0), 20) # optimizer, epochs
iceflow_trained = @time train_iceflow_UDE(H₀, UA, θ_trained, train_settings, H_refs, temp_series)
θ_trained = iceflow_trained.minimizer

data_range = -20.0:0.0
pred_A = predict_A̅(UA, θ_trained, collect(data_range)')
pred_A = [pred_A...] # flatten
true_A = A_fake(data_range) 

scatter(true_A, label="True A")
train_final = plot!(pred_A, label="Predicted A")
savefig(train_final,joinpath(root_plots,"training","final_model.png"))