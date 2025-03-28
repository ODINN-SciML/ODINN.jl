import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise
using Optimization
using EnzymeCore
using Enzyme
using ODINN
using Test
using JLD2
using Plots
using Infiltrator
using OrdinaryDiffEq
using Optim
using SciMLSensitivity
using Random
using Statistics
using Zygote
using Printf
using Lux

include("params_construction.jl")
include("grad_free_test.jl")
include("PDE_UDE_solve.jl")
include("inversion_test.jl")
include("SIA2D_adjoint.jl")
include("test_grad_loss.jl")
include("test_grad_Enzyme.jl")

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

@testset "Run all tests" begin 

# @testset "Training workflow without sensitivity analysis and AD (with MB)" grad_free_test(use_MB=false)
# @testset "Training workflow without sensitivity analysis and AD (without MB)" grad_free_test(use_MB=true)

# @testset "UDE SIA2D training with MB" ude_solve_test(; MB=true)

@testset "Parameters constructors with specified values" params_constructor_specified()

@testset "Inversion Tests" inversion_test(steady_state = true, save_refs = false)

@testset "Continuous adjoint of SIA2D" test_adjoint_SIAD2D_continuous()

@testset "Manual implementation of the backward with discrete adjoint" test_grad_discreteAdjoint()

@testset "Manual implementation of the backward with continuous adjoint" test_grad_continuousAdjoint()

@testset "Comparison of the manual backward of the loss terms with Enzyme" test_grad_loss_term()

@testset "Consistency between discrete adjoint and Enzyme AD" test_grad_Enzyme_SIAD2D()

end
