import Pkg
Pkg.activate(dirname(Base.current_project()))

if !parse(Bool, get(ENV, "CI", "false"))
    using Revise
    const printDebug = true
else
    const printDebug = false
end
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

@testset "Discrete VJP vs Enzyme VJP" test_grad_Enzyme_SIAD2D() # This test must be run first, otherwise Enzyme compilation fails because it was used before

@testset "Continuous VJP vs finite differences" test_adjoint_SIAD2D_continuous()

@testset "Implementation of discrete adjoint with discrete VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method=DiscreteVJP()); thres=[1e-2, 1e-8, 1e-2])

@testset "Implementation of discrete adjoint with continuous VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method=ContinuousVJP()); thres=[2e-2, 1e-8, 3e-2])

@testset "Implementation of continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method=DiscreteVJP()); thres=[2e-4, 1e-8, 1e-3])

# @testset "Implementation of continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method=ContinuousVJP()); thres=[2e-4, 1e-8, 1e-3])

@testset "Implementation of continuous adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method=ODINN.EnzymeVJP()); thres=[2e-4, 1e-8, 1e-3])

@testset "Backward of the loss terms vs Enzyme" test_grad_loss_term()

@testset "Implementation of continuous adjoint with discrete VJP vs Enzyme for Halfar solution" test_grad_Halfar(ContinuousAdjoint(VJP_method=DiscreteVJP()); thres=[3e-1, 1e-15, 5e-1])

@testset "Implementation of continuous adjoint with continuous VJP vs Enzyme for Halfar solution" test_grad_Halfar(ContinuousAdjoint(VJP_method=ContinuousVJP()); thres=[3e-1, 1e-15, 5e-1])

@testset "Inversion Tests" inversion_test(steady_state = true, save_refs = false)

end
