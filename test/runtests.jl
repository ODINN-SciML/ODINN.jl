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
using Random
using Statistics
using Zygote
using ProgressMeter
using Printf
using Lux
using FiniteDifferences
using JET

include("test_utils.jl")
include("params_construction.jl")
include("grad_free_test.jl")
include("SIA2D_adjoint_utils.jl")
include("inversion_test.jl")
include("SIA2D_adjoint.jl")
include("test_grad_loss.jl")
include("save_results.jl")

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
    @testset "VJPs tests" begin
        @testset "VJP (Enzyme) of SIA2D vs finite differences" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP()); target = :A, check_method=:FiniteDifferences)
        @testset "VJP (discrete) of SIA2D vs Enzyme" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres=[5e-7, 1e-6, 5e-4], target = :A)
        @testset "VJP (discrete) of SIA2D with C>0 vs Enzyme" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres=[3e-4, 2e-4, 2e-2], target = :A, C=7e-8)
        @testset "VJP (continuous) of SIA2D vs Enzyme" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = ContinuousVJP()); target = :A)
        @testset "VJP (continuous) of SIA2D with C>0 vs Enzyme" test_adjoint_SIA2D(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres=[6e-4, 7e-4, 4e-2], target = :A, C=7e-8)
    end
end

if GROUP == "All" || GROUP == "Core3"
    @testset "Adjoints tests of SIA equation with A as target" begin
        @testset "Manual implementation of the discrete adjoint with discrete VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2])
        @testset "Manual implementation of the discrete adjoint with continuous VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2])
        @testset "Manual implementation of the continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2])
        @testset "Manual implementation of the continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2])
        @testset "Manual implementation of the continuous adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP()); thres = [5e-4, 2e-8, 5e-4])
        @testset "SciMLSensitivity adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(ODINN.SciMLSensitivityAdjoint(); thres = [5e-4, 5e-8, 5e-4])
    end

    @testset "Manual backward of the loss terms vs Enzyme" test_grad_loss_term()
    # @testset "Manual implementation of the discrete VJP vs Enzyme for Halfar solution" test_grad_Halfar(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [5e-1, 1e-15, 5e-1])
    # @testset "Manual implementation of the continuous VJP vs Enzyme for Halfar solution" test_grad_Halfar(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [5e-1, 1e-15, 7e-1])
end

if GROUP == "All" || GROUP == "Core4"
@testset "Adjoint method of SIA equation with A as target and ice velocity loss" begin
    @testset "VJP (discrete) of surface_V vs finite differences" test_adjoint_surface_V(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres=[1e-6, 1e-13, 1e-6], target = :A)
    @testset "Manual implementation of the discrete adjoint with discrete VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method = DiscreteVJP()); thres = [1e-4, 1e-7, 5e-4], loss=LossV())
    # @testset "Manual implementation of the discrete adjoint with continuous VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2], loss=LossV())
    @testset "Manual implementation of the continuous adjoint with discrete VJP vs finite differences (L2)" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2], loss=LossV())
    @testset "Manual implementation of the continuous adjoint with discrete VJP vs finite differences (Log)" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2], loss=LossV(loss=LogSum(), component=:abs))
    # @testset "Manual implementation of the continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2], loss=LossV())
    # @testset "Manual implementation of the continuous adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP()); thres = [2e-4, 1e-8, 1e-3], loss=LossV())
end
end

if GROUP == "All" || GROUP == "Core5"
@testset "Adjoint method of SIA equation with hybrid D as target" begin
    @testset "Manual implementation of the continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [7e-2, 1e-4, 6e-2], target = :D_hybrid)
    @testset "Manual implementation of the continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [7e-2, 1e-4, 6e-2], target = :D_hybrid)
end
end

if GROUP == "All" || GROUP == "Core6"
@testset "Adjoint method of SIA equation with pure D as target" begin
    @testset "Manual implementation of the continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-4, 1e-2], target = :D)
    @testset "Manual implementation of the continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-4, 2e-2], target = :D)
end
end

if GROUP == "All" || GROUP == "Core7"
@testset "Inversion test" begin
    @testset "Inversion Tests w/o MB" inversion_test(use_MB = false, multiprocessing = false)
    @testset "Inversion Tests w/ MB" inversion_test(use_MB = true, multiprocessing = false)
    @testset "Inversion Tests w/o MB w/ multiprocessing" inversion_test(use_MB = false, multiprocessing = true)
end

@testset "Save results" save_simulation_test!()
end

end
