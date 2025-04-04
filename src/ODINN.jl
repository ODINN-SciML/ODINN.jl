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

using Statistics, LinearAlgebra, Random, Polynomials
using EnzymeCore
using Enzyme
# Enzyme.API.runtimeActivity!(true) # This reduces performance but fixes AD issues
Enzyme.API.strictAliasing!(false)
using JLD2
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, Optim, OptimizationOptimJL, Optimisers, OptimizationOptimisers, LineSearches
using ComponentArrays
using ChainRules: @ignore_derivatives
using SciMLBase: NoAD, CallbackSet
using MLUtils: DataLoader
using Base: @kwdef
using Lux
using Tullio
using Infiltrator
using Plots, PlotThemes
Plots.theme(:wong2) # sets overall theme for Plots
import Pkg
using Distributed
using ProgressMeter
using Downloads
using GeoStats
using ImageFiltering
using Printf

using Zygote

# ##############################################
# ############    PARAMETERS     ###############
# ##############################################

cd(@__DIR__)
const global root_dir::String = dirname(Base.current_project())
const global root_plots::String = joinpath(root_dir, "plots")

const SYSIMAGE_DIR = joinpath(homedir(), ".ODINN")
const SYSIMAGE_PATH = joinpath(SYSIMAGE_DIR, "odinn_sysimage.so")
ENV["JULIA_DEPOT_PATH"] = joinpath(homedir(), ".julia")  # Ensure shared cache

# ##############################################
# ############  ODINN LIBRARIES  ###############
# ##############################################

include(joinpath(root_dir, "src/setup/config.jl"))
# All parameters needed for the models
include(joinpath(root_dir, "src/inverse/AdjointTypes.jl"))
include(joinpath(root_dir, "src/parameters/Hyperparameters.jl"))
include(joinpath(root_dir, "src/parameters/UDEparameters.jl"))
# Simulations
include(joinpath(root_dir, "src/simulations/training_stats/TrainingStats.jl"))
include(joinpath(root_dir, "src/simulations/functional_inversions/FunctionalInversion.jl"))
include(joinpath(root_dir, "src/simulations/inversions/Inversion.jl"))
# ML models
include(joinpath(root_dir, "src/models/machine_learning/MLmodel.jl"))
# Inversion 
include(joinpath(root_dir, "src/inverse/SIA2D_adjoint.jl"))
include(joinpath(root_dir, "src/inverse/gradient.jl"))

end # module

