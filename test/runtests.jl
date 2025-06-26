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
using ProgressMeter
using Printf
using Lux
using FiniteDifferences

include("test_utils.jl")
include("params_construction.jl")
include("grad_free_test.jl")
include("SIA2D_adjoint_utils.jl")
include("PDE_UDE_solve.jl")
include("inversion_test.jl")
include("SIA2D_adjoint.jl")
include("test_grad_loss.jl")
include("test_grad_Enzyme.jl")
include("save_results.jl")

# # Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"] = "nul"

@testset "Run all tests" begin

@testset "Training workflow without sensitivity analysis and AD (without MB)" grad_free_test(use_MB = false)
@testset "Training workflow without sensitivity analysis and AD (with MB)" grad_free_test(use_MB = true)
# @testset "UDE SIA2D training with MB" ude_solve_test(; MB = true)
# @testset "Parameters constructors with specified values" params_constructor_specified()

@testset "Adjoint of unit operations inside SIA2D" begin
    @testset "Adjoint of diff" test_adjoint_diff()
    @testset "Adjoint of clamp_borders" test_adjoint_clamp_borders()
    @testset "Adjoint of avg" test_adjoint_avg()
end

@testset "Adjoint method of SIA equation with A as target" begin
    # @testset "VJP (Enzyme) of SIA2D vs finite differences" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP()); target = :A) # This test must be run first, otherwise Enzyme compilation fails because it was used before
    @testset "VJP (discrete) of SIA2D vs finite differences" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = DiscreteVJP()); target = :A)
    @testset "VJP (discrete) of SIA2D with C>0 vs finite differences" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = DiscreteVJP()); target = :A, C=7e-8)
    @testset "VJP (continuous) of SIA2D vs finite differences" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = ContinuousVJP()); target = :A)
    @testset "VJP (continuous) of SIA2D with C>0 vs finite differences" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres=[5e-4, 7e-4, 4e-2], target = :A, C=7e-8)
    @testset "Manual implementation of the discrete adjoint with discrete VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2])
    @testset "Manual implementation of the discrete adjoint with continuous VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2])
    @testset "Manual implementation of the continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2])
    @testset "Manual implementation of the continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2])
    # @testset "Manual implementation of the continuous adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP()); thres = [2e-4, 1e-8, 1e-3])
    @testset "Manual backward of the loss terms vs Enzyme" test_grad_loss_term()
    # @testset "Manual implementation of the discrete VJP vs Enzyme for Halfar solution" test_grad_Halfar(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [5e-1, 1e-15, 5e-1])
    # @testset "Manual implementation of the continuous VJP vs Enzyme for Halfar solution" test_grad_Halfar(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [5e-1, 1e-15, 7e-1])
end

@testset "Adjoint method of SIA equation with hybrid D as target" begin
    @testset "Manual implementation of the continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [7e-2, 1e-4, 6e-2], target = :D_hybrid)
    @testset "Manual implementation of the continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [7e-2, 1e-4, 6e-2], target = :D_hybrid)
end

@testset "Adjoint method of SIA equation with pure D as target" begin
    @testset "Manual implementation of the continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [7e-2, 1e-4, 6e-2], target = :D)
    @testset "Manual implementation of the continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [7e-2, 1e-4, 6e-2], target = :D)
end

@testset "Inversion test" begin
    @testset "Inversion Tests (without MB)" inversion_test(use_MB = false, multiprocessing = false)
    # @testset "Inversion Tests (with MB)" inversion_test(use_MB = true, multiprocessing = false)
end

@testset "Save results" save_simulation_test!()

end
