## Environment and packages
import Pkg
Pkg.activate(dirname(Base.current_project())) # activate project

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
using JLD2
using OrdinaryDiffEq
using DiffEqFlux
using Flux
using Tullio
using RecursiveArrayTools
using Infiltrator
using Plots
using ProgressMeter

include("helpers/parameters.jl")

#######################################################################################################
#############################             Glacier Setup            ####################################
#######################################################################################################

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
(@isdefined H???) || (const H??? = copy(argentiere.thick[:,:,1]))

# Spatial and temporal differentials
const ??x, ??y = 50, 50 #m (??x = ??y)

end # @everywhere

#??Include all functions
include("helpers/iceflow.jl")

#######################################################################################################
#############################           Reference dataset          ####################################
#######################################################################################################

(@isdefined temp_series) || (const temp_series, norm_temp_series = fake_temp_series(t???))

if create_ref_dataset 
    println("Generating reference dataset for training...")
  
    # Compute reference dataset in parallel
    @everywhere solver = Ralston()
    H_refs, V??x_refs, V??y_refs = generate_ref_dataset(temp_series, H???)
    
    ## Save reference plots
    # for (temps, H) in zip(temp_series, H_refs)
        
    #     ###  Glacier ice thickness difference  ###
    #     lim = maximum( abs.(H[end] .- H???) )
    #     hm = heatmap(H[end] .- H???, c = cgrad(:balance,rev=true), aspect_ratio=:equal,
    #         clim = (-lim, lim),
    #         title="Variation in ice thickness")

    #     tempn = floor(mean(temps))
    #     println("Saving reference_$tempn.png...")
    #     savefig(hm,joinpath(root_dir,"plots/references","reference_$tempn.png"))
    # end
        
    println("Saving reference data")
    jldsave(joinpath(root_dir, "data/PDE_refs.jld2"); H_refs, V??x_refs, V??y_refs)

end

# Load stored PDE reference datasets
PDE_refs = load(joinpath(root_dir, "data/PDE_refs.jld2"))
    
#######################################################################################################
#############################             Train UDE            ########################################
#######################################################################################################

UA = FastChain(
        FastDense(1,3, x->softplus.(x)),
        FastDense(3,10, x->softplus.(x)),
        FastDense(10,3, x->softplus.(x)),
        FastDense(3,1, sigmoid_A)
    )
    
?? = initial_params(UA)
current_epoch = 1
batch_size = length(temp_series)

cd(@__DIR__)
const root_plots = cd(pwd, "../../plots")
# Train iceflow UDE in parallel
# First train with ADAM to move the parameters into a favourable space
@everywhere solver = ROCK4()
train_settings = (ADAM(0.05), 20) # optimizer, epochs
iceflow_trained = @time train_iceflow_UDE(H???, UA, ??, train_settings, PDE_refs, temp_series)
??_trained = iceflow_trained.minimizer

# Continue training with a smaller learning rate
# train_settings = (ADAM(0.001), 20) # optimizer, epochs
# iceflow_trained = @time train_iceflow_UDE(H???, UA, ??, train_settings, H_refs, temp_series)
# ??_trained = iceflow_trained.minimizer

# Continue training with BFGS
train_settings = (BFGS(initial_stepnorm=0.02f0), 20) # optimizer, epochs
iceflow_trained = @time train_iceflow_UDE(H???, UA, ??_trained, train_settings, PDE_refs, temp_series)
??_trained = iceflow_trained.minimizer

# Save trained NN weights
save(joinpath(root_dir, "data/trained_weights.jld"), "??_trained", ??_trained)

# Plot the final trained model
data_range = -20.0:0.0
pred_A = predict_A??(UA, ??_trained, collect(data_range)')
pred_A = [pred_A...] # flatten
true_A = A_fake(data_range) 

scatter(true_A, label="True A")
train_final = plot!(pred_A, label="Predicted A")
savefig(train_final,joinpath(root_plots,"training","final_model.png"))