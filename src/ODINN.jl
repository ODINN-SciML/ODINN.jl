__precompile__() # this module is safe to precompile
module ODINN

###############################################
############       PACKAGES     ##############
###############################################

using Statistics, LinearAlgebra, Random, Polynomials
using JLD2
using OrdinaryDiffEq
using SciMLSensitivity, Optimization
using Zygote: @ignore
using Flux
using Tullio, RecursiveArrayTools
using Infiltrator
using Plots, PlotThemes
theme(:wong2) # sets overall theme for Plots
using Makie, CairoMakie
import Pkg
using Distributed
using ProgressMeter
using PyCall

###############################################
#############    PARAMETERS     ###############
###############################################

include("helpers/parameters.jl")

###############################################
#############  ODINN LIBRARIES  ###############
###############################################

cd(@__DIR__)
global root_dir = dirname(Base.current_project())
global root_plots = joinpath(root_dir, "plots")

include("helpers/utils.jl")

#### Plotting functions  ###
include("helpers/plotting.jl")
### Iceflow modelling functions  ###
# (includes utils.jl as well)
include("helpers/iceflow.jl")
### Mass balance modelling functions ###
include("helpers/mass_balance.jl")

###############################################
#############  PYTHON LIBRARIES  ##############
###############################################

const netCDF4 = PyNULL()
const cfg = PyNULL()
const utils = PyNULL()
const workflow = PyNULL()
const tasks = PyNULL()
const graphics = PyNULL()
const bedtopo = PyNULL()
const MBsandbox = PyNULL()

# Essential Python libraries
const np = PyNULL()
const xr = PyNULL()

###############################################
######### PYTHON JULIA INTERACTIONS  ##########
###############################################

include(joinpath(ODINN.root_dir, "src/helpers/config.jl"))
### Climate data processing  ###
include(joinpath(ODINN.root_dir, "src/helpers/climate.jl"))
### OGGM configuration settings  ###
include(joinpath(ODINN.root_dir, "src/helpers/oggm.jl"))

end # module

