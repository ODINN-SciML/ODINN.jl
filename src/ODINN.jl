__precompile__() # this module is safe to precompile
module ODINN

# ##############################################
# ###########       PACKAGES     ##############
# ##############################################

using Statistics, LinearAlgebra, Random, Polynomials
using JLD2
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, Optim, OptimizationOptimJL
using IterTools: ncycle
using ChainRules: @ignore_derivatives
using Base: @kwdef
using Flux
using Tullio
using Infiltrator
using Plots, PlotThemes
Plots.theme(:wong2) # sets overall theme for Plots
using CairoMakie, GeoMakie
import Pkg
using Distributed
using ProgressMeter
using PyCall
using Downloads
using SnoopPrecompile, TimerOutputs

# ##############################################
# ############    PARAMETERS     ###############
# ##############################################

cd(@__DIR__)
const global root_dir::String = dirname(Base.current_project())
const global root_plots::String = joinpath(root_dir, "plots")


# ##############################################
# ############  PYTHON LIBRARIES  ##############
# ##############################################

@precompile_setup begin

const netCDF4::PyObject = PyNULL()
const cfg::PyObject = PyNULL()
const utils::PyObject = PyNULL()
const workflow::PyObject = PyNULL()
const tasks::PyObject = PyNULL()
const global_tasks::PyObject = PyNULL()
const graphics::PyObject = PyNULL()
const bedtopo::PyObject = PyNULL()
const millan22::PyObject = PyNULL()
const MBsandbox::PyObject = PyNULL()
const salem::PyObject = PyNULL()

# Essential Python libraries
const np::PyObject = PyNULL()
const xr::PyObject = PyNULL()
const rioxarray::PyObject = PyNULL()
const pd::PyObject = PyNULL()

# ##############################################
# ############  ODINN LIBRARIES  ###############
# ##############################################

@precompile_all_calls begin

include(joinpath(ODINN.root_dir, "src/setup/config.jl"))
# All parameters needed for the models
include(joinpath(ODINN.root_dir, "src/parameters/Parameters.jl"))
# Anything related to managing glacier topographical and climate data
include(joinpath(ODINN.root_dir, "src/glaciers/Glacier.jl"))
# All structures and functions related to ODINN models
include(joinpath(ODINN.root_dir, "src/models/Model.jl"))
# Everything related to running simulations in ODINN
include(joinpath(ODINN.root_dir, "src/simulations/Simulation.jl"))

end # @precompile_setup
end # @precompile_all_calls
end # module

