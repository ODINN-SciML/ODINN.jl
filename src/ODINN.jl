__precompile__() # this module is safe to precompile

# """
# # ODINN.jl

# ODINN is an open-source glacier evolution model and project that investigates innovative 
# hybrid methods to discover new laws governing glacier physics. 
# By leveraging differentiable programming techniques, we are developing hybrid models 
# that integrate differential equations describing ice flow with machine learning models 
# to learn and parameterize specific components of these equations. 
# This approach facilitates the discovery of parameterizations for glacier processes, 
# helping to bridge the gap between our current mechanistic understanding of glacier 
# physics and emerging observational data.
# """
module ODINN

# ##############################################
# ###########       PACKAGES     ##############
# ##############################################

# ODINN subpackages
using Reexport
@reexport using Huginn # imports Muninn and Sleipnir

import Sleipnir: Parameters, Model
using Statistics, LinearAlgebra
using Random, Distributions
using EnzymeCore
using Enzyme
using JLD2
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, Optim, OptimizationOptimJL, Optimisers, OptimizationOptimisers, LineSearches
using ComponentArrays
using ChainRules: @ignore_derivatives
using SciMLBase: NoAD, CallbackSet
using DiffEqCallbacks: PeriodicCallback
using MLUtils: DataLoader
using Base: @kwdef
using Lux
using Tullio
using Infiltrator
using Plots, PlotThemes, PlotlyJS
Plots.theme(:wong2) # sets overall theme for Plots
import Pkg
using Distributed
using ProgressMeter
using Downloads
using ImageFiltering
using Printf
using Interpolations, GeoStats
using FastGaussQuadrature
using Zygote
using TensorBoardLogger
using Dates
using MLStyle

# ##############################################
# ############    PARAMETERS     ###############
# ##############################################

cd(@__DIR__)
const global root_dir::String = dirname(Base.current_project())
const global root_plots::String = joinpath(root_dir, "plots")

# const SYSIMAGE_DIR = joinpath(homedir(), ".ODINN")
# const SYSIMAGE_PATH = joinpath(SYSIMAGE_DIR, "odinn_sysimage.so")
# ENV["JULIA_DEPOT_PATH"] = joinpath(homedir(), ".julia")  # Ensure shared cache

# ##############################################
# ############  ODINN LIBRARIES  ###############
# ##############################################

include(joinpath(root_dir, "src/setup/config.jl"))
# Losses
include(joinpath(root_dir, "src/losses/Losses.jl"))
# All parameters needed for the models
include(joinpath(root_dir, "src/inverse/VJPTypes.jl"))
include(joinpath(root_dir, "src/inverse/AdjointTypes.jl"))
include(joinpath(root_dir, "src/parameters/Hyperparameters.jl"))
include(joinpath(root_dir, "src/parameters/UDEparameters.jl"))
# Simulations
include(joinpath(root_dir, "src/simulations/results/Results.jl"))
include(joinpath(root_dir, "src/simulations/functional_inversions/FunctionalInversion.jl"))
include(joinpath(root_dir, "src/simulations/inversions/Inversion.jl"))
# ML models
include(joinpath(root_dir, "src/models/machine_learning/ML_utils.jl"))
include(joinpath(root_dir, "src/models/machine_learning/NN_utils.jl"))
include(joinpath(root_dir, "src/models/target/Target.jl"))
include(joinpath(root_dir, "src/models/machine_learning/MLmodel.jl"))
# Parameterizations
include(joinpath(root_dir, "src/laws/Laws.jl"))

# Inversion for SIA equation
include(joinpath(root_dir, "src/inverse/SIA2D/Inversion.jl"))

# Results
include(joinpath(root_dir, "src/results/TrainingResults.jl"))

end # module
