import Pkg
Pkg.activate(dirname(Base.current_project()))

# Use a fork of SciMLSensitivity until https://github.com/SciML/SciMLSensitivity.jl/issues/1238 is fixed
Pkg.develop(url="https://github.com/albangossard/SciMLSensitivity.jl/")

const GROUP = get(ENV, "GROUP", "All")
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
using Statistics
using Zygote
using ProgressMeter
using Printf
using Lux
using FiniteDifferences
using JET
using MLStyle
import DifferentiationInterface as DI

include("test_utils.jl")
include("params_construction.jl")
include("grad_free_test.jl")
include("SIA2D_adjoint_utils.jl")
include("inversion_test.jl")
include("SIA2D_adjoint.jl")
include("MB_VJP.jl")
include("test_grad_loss.jl")
include("save_results.jl")

# Set random seed
using Random
Random.seed!(1234)

# # Activate to avoid GKS backend Plot issues in the JupyterHub
ENV["GKSwstype"] = "nul"

@info "Running group $(GROUP)"

@testset "Run all tests" begin

if GROUP == "All" || GROUP == "Core1"
@testset "Training workflow without sensitivity analysis and AD (without MB)" grad_free_test(use_MB = false)
@testset "Training workflow without sensitivity analysis and AD (with MB)" grad_free_test(use_MB = true)
@testset "Parameters constructors with specified values" params_constructor_specified()

@testset "Adjoint of unit operations inside SIA2D" begin
    @testset "Adjoint of diff" test_adjoint_diff()
    @testset "Adjoint of clamp_borders" test_adjoint_clamp_borders()
    @testset "Adjoint of avg" test_adjoint_avg()
end
end

if GROUP == "All" || GROUP == "Core2"
    @testset "VJPs tests with A as target" begin
        @testset "VJP (Enzyme) of MB vs finite differences" test_MB_VJP(ODINN.EnzymeVJP())
        @testset "VJP (discrete) of MB vs finite differences" test_MB_VJP(DiscreteVJP())
        @testset "VJP (Enzyme) of SIA2D vs finite differences" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP()); target = :A)
        @testset "VJP (discrete) of SIA2D vs finite differences" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres=[5e-7, 1e-6, 5e-4], target = :A)
        @testset "VJP (discrete) of SIA2D with C>0 vs finite differences" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres=[3e-4, 2e-4, 2e-2], target = :A, C=7e-8)
        @testset "VJP (continuous) of SIA2D vs finite differences" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = ContinuousVJP()); target = :A)
        @testset "VJP (continuous) of SIA2D with C>0 vs finite differences" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres=[6e-4, 7e-4, 4e-2], target = :A, C=7e-8)
    end

    @testset "Manual backward of the loss terms vs Enzyme" begin
        @testset "L2Sum" test_grad_L2Sum()
        @testset "TikhonovRegularization" test_grad_TikhonovRegularization()
    end
end

if GROUP == "All" || GROUP == "Core3"
    @testset "Adjoints tests of SIA equation with A as target" begin
        @testset "Discrete adjoint with discrete VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2])
        @testset "Discrete adjoint with discrete VJP vs finite differences for classical inversions" test_grad_finite_diff(DiscreteAdjoint(VJP_method = DiscreteVJP()); functional_inv = false, thres = [1e-2, 1e-5, 1e-2])
        @testset "Discrete adjoint with discrete VJP vs finite differences (initial condition)" test_grad_finite_diff(DiscreteAdjoint(VJP_method = DiscreteVJP()); thres = [3e-2, 8e-5, 3e-2], train_initial_conditions = true)
        @testset "Discrete adjoint with continuous VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2])
        @testset "Continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2])
        @testset "Continuous adjoint with discrete VJP vs finite differences (initial condition)" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2],  train_initial_conditions = true)
        @testset "Continuous adjoint with discrete VJP vs finite differences w/ Enzyme MB VJP" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP(regressorADBackend = DI.AutoZygote()), MB_VJP = ODINN.EnzymeVJP()); thres = [2e-3, 2e-5, 2e-3], use_MB = true) # This test uses Zygote for the differentiation of the laws because Mooncake has to store modules inside the VJPsPrepLaw struct which is not compatible with Enzyme.make_zero
        @testset "Continuous adjoint with discrete VJP vs finite differences w/ discrete MB VJP" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = DiscreteVJP()); thres = [2e-2, 2e-5, 2e-2], use_MB = true)
        @testset "Continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2])
        @testset "Continuous adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP()); thres = [5e-4, 2e-8, 5e-4])
        @testset "SciMLSensitivity adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(ODINN.SciMLSensitivityAdjoint(); thres = [5e-4, 5e-8, 5e-4])
    end

    # @testset "Manual implementation of the discrete VJP vs Enzyme for Halfar solution" test_grad_Halfar(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [5e-1, 1e-15, 5e-1])
    # @testset "Manual implementation of the continuous VJP vs Enzyme for Halfar solution" test_grad_Halfar(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [5e-1, 1e-15, 7e-1])
end

if GROUP == "All" || GROUP == "Core4"
@testset "Adjoint method of SIA equation with A as target and ice velocity loss" begin
    @testset "VJP (discrete) of surface_V vs finite differences" test_adjoint_surface_V(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres=[1e-6, 1e-13, 1e-6], target = :A)
    @testset "Discrete adjoint with discrete VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method = DiscreteVJP()); thres = [1e-4, 1e-7, 5e-4], loss=LossV())
    # @testset "Discrete adjoint with continuous VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2], loss=LossV())
    @testset "Continuous adjoint with discrete VJP vs finite differences (L2)" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2], loss=LossV())
    @testset "Continuous adjoint with discrete VJP vs finite differences (Log)" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2], loss=LossV(loss=LogSum(), component=:abs))
    # @testset "Continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2], loss=LossV())
    # @testset "Continuous adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP()); thres = [2e-4, 1e-8, 1e-3], loss=LossV())
end
end

if GROUP == "All" || GROUP == "Core5"
@testset "Adjoint method of SIA equation with hybrid D as target" begin
    @testset "Continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [7e-2, 1e-4, 6e-2], target = :D_hybrid)
    @testset "Continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [7e-2, 1e-4, 6e-2], target = :D_hybrid)
end
end

if GROUP == "All" || GROUP == "Core6"
@testset "Adjoint method of SIA equation with pure D as target" begin
    @testset "Continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-4, 1e-2], target = :D)
    @testset "Continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-4, 2e-2], target = :D)
end
end

if GROUP == "All" || GROUP == "Core7"
    @testset "Multi-objective function and regularization test" begin
        @testset "Gradient evaluation testing MultiLoss API" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2], loss=MultiLoss(losses=(LossH(),), λs=(0.4,)))
        @testset "Gradient evaluation testing just regularization" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [3e-2, 1e-5, 3e-2], loss=MultiLoss(losses=(VelocityRegularization(),), λs=(1e2,)))
        @testset "Gradient evaluation testing empirical and regularization" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2], loss=MultiLoss(losses=(LossH(), VelocityRegularization()), λs=(1e-2, 2e-1)))
    end
end

if GROUP == "All" || GROUP == "Core8"
    @testset "Classical inversions" begin
        @testset "Classical inversion w/o MB" inversion_test(use_MB = false, multiprocessing = false, functional_inv = false)
    end
    @testset "Functional inversions" begin
        @testset "Functional inversion w/o MB" inversion_test(use_MB = false, multiprocessing = false)
        @testset "Functional inversion w/ MB" inversion_test(use_MB = true, multiprocessing = false, grad = ContinuousAdjoint(VJP_method = DiscreteVJP(regressorADBackend = DI.AutoZygote())))
        @testset "Functional inversion w/o MB w/ multiprocessing" inversion_test(use_MB = false, multiprocessing = true)
    end
end

if GROUP == "All" || GROUP == "Core9"
@testset "Multiglacier inversion test" begin
    @testset "Continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2], multiglacier = true)
    @testset "Continuous adjoint with discrete VJP vs finite differences (initial condition)" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2], multiglacier = true, train_initial_conditions = true)
end
end

if GROUP == "All" || GROUP == "Core10"
@testset "Save results" begin
    @testset "Single glacier" save_simulation_test!(multiglacier = false)
    @testset "Multiple glaciers" save_simulation_test!(multiglacier = true)
end
end

end
