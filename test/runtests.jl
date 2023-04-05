import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise
using ODINN
using Test
using JLD2
using Plots
using Infiltrator

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

atol = 0.1

@testset "PDE and UDE SIA solvers" begin include("PDE_UDE_solve.jl") end
# @testset "SIA UDE training" begin include("UDE_train.jl") end

