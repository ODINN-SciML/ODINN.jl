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
using JLD2
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, Optim, OptimizationOptimJL, Optimisers, OptimizationOptimisers
using IterTools: ncycle
using Zygote
using ChainRules: @ignore_derivatives
using SciMLBase: NoAD, CallbackSet
using Base: @kwdef
using Flux
using Tullio
using Infiltrator, Cthulhu
using Plots, PlotThemes
Plots.theme(:wong2) # sets overall theme for Plots
import Pkg
using Distributed
using ProgressMeter
using Downloads
using TimerOutputs
using GeoStats
using ImageFiltering
using Printf

# using Enzyme
# Enzyme.API.runtimeActivity!(true)

# ##############################################
# ############    PARAMETERS     ###############
# ##############################################

cd(@__DIR__)
const global root_dir::String = dirname(Base.current_project())
const global root_plots::String = joinpath(root_dir, "plots")


# ##############################################
# ############  ODINN LIBRARIES  ###############
# ##############################################

include(joinpath(root_dir, "src/setup/config.jl"))
# All parameters needed for the models
include(joinpath(root_dir, "src/parameters/Hyperparameters.jl"))
include(joinpath(root_dir, "src/parameters/UDEparameters.jl"))
# ML models
include(joinpath(root_dir, "src/models/machine_learning/MLmodel.jl"))
# Simulations
include(joinpath(root_dir, "src/simulations/training_stats/TrainingStats.jl"))
include(joinpath(root_dir, "src/simulations/functional_inversions/FunctionalInversion.jl"))
include(joinpath(root_dir, "src/simulations/inversions/Inversion.jl"))

end # module

