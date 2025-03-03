import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise
using Optimization
using EnzymeCore
using Enzyme
# Enzyme.API.runtimeActivity!(true) # This reduces performance but fixes AD issues
Enzyme.API.strictAliasing!(false)
using ODINN
using Test
using JLD2
using Plots
using Infiltrator
using OrdinaryDiffEq
using Optim
using SciMLSensitivity

include("params_construction.jl")
include("grad_free_test.jl")
include("PDE_UDE_solve.jl")
include("inversion_test.jl")

# Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"]="nul"

@testset "Run all tests" begin 

@testset "Training workflow without sensitivity analysis and AD (with MB)" grad_free_test(use_MB=false)
@testset "Training workflow without sensitivity analysis and AD (without MB)" grad_free_test(use_MB=true)

atol = 2.0
@testset "UDE SIA2D training with MB" ude_solve_test(atol; MB=true)

@testset "Parameters constructors with specified values" params_constructor_specified()

@testset "Inversion Tests" inversion_test(steady_state = true, save_refs = false)

end