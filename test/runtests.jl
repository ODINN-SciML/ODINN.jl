import Pkg
Pkg.activate(dirname(Base.current_project()))

using PyCall
# Update SSL certificate to avoid issue in GitHub Actions CI
certifi = pyimport("certifi")
ENV["SSL_CERT_FILE"] = certifi.where()
# println("Current SSL certificate: ", ENV["SSL_CERT_FILE"])

using Revise
using ODINN
using Test
using JLD2
using Plots
using Infiltrator

include("PDE_UDE_solve.jl")

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

atol = 0.01
@testset "PDE and UDE SIA solvers without MB" pde_solve_test(atol; MB=false, fast=true)

atol = 2.0
@testset "PDE and UDE SIA solvers with MB" pde_solve_test(atol; MB=true, fast=true)

# @testset "SIA UDE training" begin include("UDE_train.jl") end

