__precompile__() # this module is safe to precompile
module ODINN

###############################################
############       PACKAGES     ##############
###############################################

using Statistics, LinearAlgebra, Random, Polynomials
using JLD2
using OrdinaryDiffEq, DiffEqCallbacks
using SciMLSensitivity, Optimization
using Zygote: @ignore, Buffer 
using Flux
using Tullio, RecursiveArrayTools
using Infiltrator
using Plots, PlotThemes
Plots.theme(:wong2) # sets overall theme for Plots
using Makie, CairoMakie, GeoMakie
import Pkg
using Distributed, ParallelDataTransfer
using ProgressMeter
using PyCall
using HDF5
using SnoopPrecompile 

###############################################
#############    PARAMETERS     ###############
###############################################

@precompile_setup begin

include("helpers/parameters.jl")

###############################################
#############  ODINN LIBRARIES  ###############
###############################################

cd(@__DIR__)
global root_dir = dirname(Base.current_project())
global root_plots = joinpath(root_dir, "plots")

@precompile_all_calls begin

include("helpers/utils.jl")
# Functions to retrieve data for the simulation's initial conditions
include("helpers/initial_conditions.jl")
#### Plotting functions  ###
include("helpers/plotting.jl")
### Iceflow modelling functions  ###
# (includes utils.jl as well)
include("helpers/iceflow.jl")
### Mass balance modelling functions ###
include("helpers/mass_balance.jl")
### Ice rheology inversion functions ###
include("helpers/inversions.jl")

end # @precompile_setup
end # @precompile_all_calls



###############################################
#############  PYTHON LIBRARIES  ##############
###############################################

@precompile_setup begin

const netCDF4 = PyNULL()
const cfg = PyNULL()
const utils = PyNULL()
const workflow = PyNULL()
const tasks = PyNULL()
const graphics = PyNULL()
const bedtopo = PyNULL()
const millan22 = PyNULL()
const MBsandbox = PyNULL()

# Essential Python libraries
const np = PyNULL()
const xr = PyNULL()

###############################################
######### PYTHON JULIA INTERACTIONS  ##########
###############################################

@precompile_all_calls begin

include(joinpath(ODINN.root_dir, "src/helpers/config.jl"))
### Climate data processing  ###
include(joinpath(ODINN.root_dir, "src/helpers/climate.jl"))
### OGGM configuration settings  ###
include(joinpath(ODINN.root_dir, "src/helpers/oggm.jl"))

end # @precompile_setup
end # @precompile_all_calls
end # module

