__precompile__() # this module is safe to precompile
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
using Base: @kwdef
using Flux
using Tullio
using Infiltrator, Cthulhu
using Plots, PlotThemes
Plots.theme(:wong2) # sets overall theme for Plots
import Pkg
using Distributed
using ProgressMeter
using PyCall
using Downloads
using TimerOutputs
using GeoStats
using ImageFiltering

# ##############################################
# ############    PARAMETERS     ###############
# ##############################################

cd(@__DIR__)
const global root_dir::String = dirname(Base.current_project())
const global root_plots::String = joinpath(root_dir, "plots")


# ##############################################
# ############  PYTHON LIBRARIES  ##############
# ##############################################

# We either retrieve the reexported Python libraries from Sleipnir or we start from scratch
const netCDF4::PyObject = isdefined(Sleipnir, :netCDF4) ? Sleipnir.netCDF4 : PyNULL()
const cfg::PyObject = isdefined(Sleipnir, :cfg) ? Sleipnir.cfg : PyNULL()
const utils::PyObject = isdefined(Sleipnir, :utils) ? Sleipnir.utils : PyNULL()
const workflow::PyObject = isdefined(Sleipnir, :workflow) ? Sleipnir.workflow : PyNULL()
const tasks::PyObject = isdefined(Sleipnir, :tasks) ? Sleipnir.tasks : PyNULL()
const global_tasks::PyObject = isdefined(Sleipnir, :global_tasks) ? Sleipnir.global_tasks : PyNULL()
const graphics::PyObject = isdefined(Sleipnir, :graphics) ? Sleipnir.graphics : PyNULL()
const bedtopo::PyObject = isdefined(Sleipnir, :bedtopo) ? Sleipnir.bedtopo : PyNULL()
const millan22::PyObject = isdefined(Sleipnir, :millan22) ? Sleipnir.millan22 : PyNULL()
const MBsandbox::PyObject = isdefined(Sleipnir, :MBsandbox) ? Sleipnir.MBsandbox : PyNULL()
const salem::PyObject = isdefined(Sleipnir, :salem) ? Sleipnir.salem : PyNULL()

# Essential Python libraries
const xr::PyObject = isdefined(Sleipnir, :xr) ? Sleipnir.xr : PyNULL()
const rioxarray::PyObject = isdefined(Sleipnir, :rioxarray) ? Sleipnir.rioxarray : PyNULL()
const pd::PyObject = isdefined(Sleipnir, :pd) ? Sleipnir.pd : PyNULL()

# ##############################################
# ############  ODINN LIBRARIES  ###############
# ##############################################

include(joinpath(ODINN.root_dir, "src/setup/config.jl"))
# All parameters needed for the models
include(joinpath(ODINN.root_dir, "src/parameters/Hyperparameters.jl"))
include(joinpath(ODINN.root_dir, "src/parameters/UDEparameters.jl"))
# ML models
include(joinpath(ODINN.root_dir, "src/models/machine_learning/MLmodel.jl"))
# Simulations
include(joinpath(ODINN.root_dir, "src/simulations/functional_inversions/FunctionalInversion.jl"))
include(joinpath(ODINN.root_dir, "src/simulations/inversions/Inversion.jl"))

end # module

