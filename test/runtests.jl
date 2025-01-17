import Pkg
Pkg.activate(dirname(Base.current_project()))

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
include("inversion_test.jl")

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

atol = 2.0
@testset "UDE SIA2D training with MB" ude_solve_test(atol; MB=true)

@testset "Inversion Tests" inversion_test(steady_state = true, save_refs = false)
# @testset "SIA UDE training" begin include("UDE_train.jl") end

