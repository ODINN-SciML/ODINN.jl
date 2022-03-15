module ODINN

###############################################
##############       EXPORTS   ################
###############################################


###############################################
############  JULIA ENVIRONMENT  ##############
###############################################

using Statistics, LinearAlgebra, Random, Polynomials
using JLD2
using OrdinaryDiffEq, DiffEqFlux
using Zygote: @ignore
using Flux
using Tullio, RecursiveArrayTools
using Infiltrator
using Plots
using ParallelDataTransfer, Distributed
using Dates # to provide correct Julian time slices 
using Makie, CairoMakie
import Pkg
using Distributed
using ProgressMeter

###############################################
#############    PARAMETERS     ###############
###############################################

include("helpers/parameters.jl")

###############################################
#############  ODINN LIBRARIES  ###############
###############################################

cd(@__DIR__)
global root_dir = dirname(Base.current_project())
global root_plots = cd(joinpath(root_dir, "plots"))

#### Plotting functions  ###
include("helpers/plotting.jl")
### Iceflow modelling functions  ###
# (includes utils.jl as well)
include("helpers/iceflow.jl")

include("helpers/config.jl")

end # module

