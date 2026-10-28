import Pkg
function is_included_in_repl()
    # Handle github CI
    if get(ENV, "CI_FAST", "false")=="true"
        return true
    end
    frames = StackTraces.stacktrace()
    # Handle manual include by the user in the REPL
    for frame in frames
        if occursin("start_repl_backend", string(frame.func))
            return true
        end
    end
    return false
end

Pkg.activate(dirname(Base.current_project()))
Pkg.instantiate() # Need this to setup the ODINN env for multiprocessing
if is_included_in_repl()
    # The Project.toml of the test environment to be used when running with include is in a subfolder to avoid that Julia uses this file in test mode
    Pkg.activate(dirname(Base.current_project())*"/test/test_env/")
    Pkg.resolve()
end

const GROUP = get(ENV, "GROUP", "All")
const CI = parse(Bool, get(ENV, "CI", "false"))
if !CI
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
using Infiltrator
using OrdinaryDiffEq
using LinearAlgebra
using Optim, Optimisers, OptimizationOptimisers, OptimizationOptimJL
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
using Aqua

include("params_construction.jl")
include("grad_free_test.jl")
include("SIA2D_adjoint_utils.jl")
include("inversion_test.jl")
include("SIA2D_adjoint.jl")
include("MB_VJP.jl")
include("test_grad_loss.jl")
include("save_results.jl")
include("Aqua.jl")

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
        @testset "Inversion instantiation" test_inversion_instantiation()
        @testset "C parameterization of inversion containers" test_C_parameterization()

        @testset "Adjoint of unit operations inside SIA2D" begin
            @testset "Adjoint of diff" test_adjoint_diff()
            @testset "Adjoint of clamp_borders" test_adjoint_clamp_borders()
            @testset "Adjoint of avg" test_adjoint_avg()
            @testset "5-point Laplacian and its adjoint" test_laplacian()
            @testset "Defaults of the initial thickness and bed regularizations" test_regularization_defaults()
        end
    end

    if GROUP == "All" || GROUP == "Core2"
        @testset "VJPs tests with A as target" begin
            @testset "VJP (Enzyme) of MB vs finite differences" test_MB_VJP(ODINN.EnzymeVJP())
            @testset "VJP (discrete) of MB vs finite differences" test_MB_VJP(DiscreteVJP())
            @testset "VJP (Enzyme) of SIA2D vs finite differences" test_adjoint_SIA2D(
                ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP()); target = :A)
            @testset "VJP (discrete) of SIA2D vs finite differences" test_adjoint_SIA2D(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [5e-7, 1e-6, 5e-4], target = :A)
            @testset "VJP (discrete) of SIA2D with C>0 vs finite differences" test_adjoint_SIA2D(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [3e-4, 2e-4, 2e-2], target = :A, C = 7e-8)
            @testset "VJP (continuous) of SIA2D vs finite differences" test_adjoint_SIA2D(
                ContinuousAdjoint(VJP_method = ContinuousVJP()); target = :A)
            @testset "VJP (continuous) of SIA2D with C>0 vs finite differences" test_adjoint_SIA2D(
                ContinuousAdjoint(VJP_method = ContinuousVJP());
                thres = [6e-4, 7e-4, 4e-2], target = :A, C = 7e-8)
            @testset "VJP (discrete) of SIA2D with classical scalar inversion vs finite differences" test_adjoint_SIA2D(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [6e-4, 7e-4, 4e-2], target = :A, functional_inv = false)
            @testset "VJP (discrete) of SIA2D with classical gridded inversion vs finite differences" test_adjoint_SIA2D(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [6e-4, 7e-4, 4e-2],
                target = :A, functional_inv = false, scalar = false)
        end

        @testset "Manual backward of the loss terms vs Enzyme" begin
            @testset "L2Sum" test_grad_L2Sum()
            @testset "TikhonovRegularization" test_grad_TikhonovRegularization()
            @testset "V magnitude chain rule (:abs)" test_grad_V_from_Vxy()
            @testset "Initial condition filters are type stable" test_initial_condition_filter_type_stability()
            @testset "LossAvgV time window guard" test_loss_time_window_guard()
            @testset "Observation weights" test_observation_weights()
            @testset "The first observation counts" test_first_observation_counts()
            @testset "Infinite loss when the forward simulation fails" test_loss_or_inf()
            @testset "Mean of (misfit / σ)²" test_loss_sigma()
        end
    end

    if GROUP == "All" || GROUP == "Core3"
        @testset "Manual adjoint methods of SIA equation with A as target" begin
            @testset "Discrete adjoint with discrete VJP vs finite differences" test_grad_finite_diff(
                DiscreteAdjoint(VJP_method = DiscreteVJP()); thres = [5e-3, 1e-8, 5e-3])
            @testset "Discrete adjoint with discrete VJP vs finite differences for scalar classical inversions" test_grad_finite_diff(
                DiscreteAdjoint(VJP_method = DiscreteVJP());
                functional_inv = false, thres = [5e-3, 1e-8, 5e-3])
            # θ.IC is scaled by H₀_scale, so the IC part dominates the norm. The angle was measured
            # at 2.2e-6, and the discrete adjoint is not accurate for H₀, so this is a sanity check
            @testset "Discrete adjoint with discrete VJP vs finite differences (initial condition)" test_grad_finite_diff(
                DiscreteAdjoint(VJP_method = DiscreteVJP());
                thres = [5e-3, 1e-5, 5e-3], train_initial_conditions = true)
            # ContinuousVJP is the derivative of the continuous equation, not of the discrete stencil: a σ per
            # cell puts the weight on the thin margin cells where they differ most (ratio 1.9e-3), so use one σ
            @testset "Discrete adjoint with continuous VJP vs finite differences" test_grad_finite_diff(
                DiscreteAdjoint(VJP_method = ContinuousVJP()); thres = [1e-4, 1e-8, 1e-4],
                loss = LossH(σ = 20.0))
            @testset "Continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-3, 1e-8, 1e-3])
            @testset "Continuous adjoint with discrete VJP vs finite differences (one thickness survey)" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-3, 1e-8, 1e-3],
                single_survey = true)
            @testset "Continuous adjoint with discrete VJP vs finite differences (initial condition)" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [5e-4, 1e-8, 5e-4], train_initial_conditions = true)
            # These shrink A so the gradient is mass balance dominated, which also makes it
            # ~1e-3 rather than ~1. A finite difference has to be measured accordingly: with
            # an adaptive solver the loss is discontinuous in θ, and a step suited to a
            # gradient of order one sits deep in the cancellation regime here. Sweeping it
            # (GROUP = "Core3MBfd") shows the difference converging onto the adjoint —
            # 4.8e-1 off at 1e-9 on the smallest component, 2.1e-4 at 1e-5.
            mb_fd = (adaptive = false, dt = 1.0/240.0, fd_delta = 1e-5, thres_fd = 5e-3)
            @testset "Continuous adjoint with discrete VJP vs finite differences w/ Enzyme MB VJP" test_grad_finite_diff(
                ContinuousAdjoint(
                    VJP_method = DiscreteVJP(regressorADBackend = DI.AutoZygote()),
                    MB_VJP = ODINN.EnzymeVJP());
                thres = [3e-3, 1e-8, 3e-3],
                use_MB = true, mb_fd...) # This test uses Zygote for the differentiation of the laws because Mooncake has to store modules inside the VJPsPrepLaw struct which is not compatible with Enzyme.make_zero
            @testset "Continuous adjoint with discrete VJP vs finite differences w/ discrete MB VJP" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = DiscreteVJP());
                thres = [3e-3, 1e-8, 3e-3], use_MB = true, mb_fd...)
            # A nonzero temp_bias moves the PDD/snow clamp thresholds, which the MB VJPs
            # must pick up from the MB model rather than assume away.
            @testset "Continuous adjoint w/ discrete MB VJP and temperature bias" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = DiscreteVJP());
                thres = [3e-3, 1e-8, 3e-3], use_MB = true, temp_bias = 1.0, mb_fd...)
            @testset "Continuous adjoint w/ Enzyme MB VJP and temperature bias" test_grad_finite_diff(
                ContinuousAdjoint(
                    VJP_method = DiscreteVJP(regressorADBackend = DI.AutoZygote()),
                    MB_VJP = ODINN.EnzymeVJP());
                thres = [3e-3, 1e-8, 3e-3], use_MB = true, temp_bias = 1.0, mb_fd...)
            # Calibration turns mass_balance into one model per glacier, so this covers the
            # vector of MB models flowing through the VJPs.
            @testset "Continuous adjoint w/ discrete MB VJP and calibrated MB" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = DiscreteVJP());
                thres = [2e-3, 1e-10, 2e-3], use_MB = true, calibrate_MB = true, mb_fd...)
            @testset "Continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [
                    5e-3, 1e-10, 5e-3])
            @testset "Continuous adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP());
                thres = [5e-4, 1e-10, 5e-4])
            @testset "SciMLSensitivity adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(
                ODINN.SciMLSensitivityAdjoint(); thres = [1e-5, 1e-13, 1e-5])
            @testset "SciMLSensitivity auto-adjoint vs manual ContinuousAdjoint for LawA" test_grad_sciml_vs_manual(thres = [
                1e-3, 1e-13, 1e-3])
        end

        # @testset "Manual implementation of the discrete VJP vs Enzyme for Halfar solution" test_grad_Halfar(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [5e-1, 1e-15, 5e-1])
        # @testset "Manual implementation of the continuous VJP vs Enzyme for Halfar solution" test_grad_Halfar(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [5e-1, 1e-15, 7e-1])
    end

    if GROUP == "All" || GROUP == "Core4"
        @testset "Manual adjoint methods of SIA equation with A as target and ice velocity loss" begin
            @testset "VJP (discrete) of surface_V vs finite differences" test_adjoint_surface_V(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [1e-6, 1e-13, 1e-6], target = :A)
            @testset "VJP (discrete) of surface_V with sliding vs finite differences" test_adjoint_surface_V(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [1e-6, 1e-13, 1e-6], target = :A, sliding = true)
            @testset "Discrete adjoint with discrete VJP vs finite differences" test_grad_finite_diff(
                DiscreteAdjoint(VJP_method = DiscreteVJP());
                thres = [1e-4, 1e-7, 5e-4], loss = LossV())
            # @testset "Discrete adjoint with continuous VJP vs finite differences" test_grad_finite_diff(DiscreteAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2], loss=LossV())
            @testset "Continuous adjoint with discrete VJP vs finite differences (L2)" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [1e-4, 1e-5, 1e-4], loss = LossV())
            @testset "Continuous adjoint with discrete VJP vs finite differences (Log)" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-4, 1e-5, 1e-4],
                loss = LossV(loss = LogSum(), component = :abs))
            # @testset "Continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2], loss=LossV())
            # @testset "Continuous adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP()); thres = [2e-4, 1e-8, 1e-3], loss=LossV())

            # Velocity losses with the automatic adjoint. `batch_loss_iceflow_transient` used
            # to read θ off the `InversionBinder`, so Zygote dropped the part of the gradient
            # that flows through the ODE, and the loss value stayed exact. `LossH` cannot catch
            # this (it never uses θ), so the guard must be a velocity loss: without the fix, the
            # first two cells below are 14.8x and 1.05x wrong.
            # Fixed step and a pinned solver: the harness picks a different integrator per
            # adjoint otherwise, and an adaptive step makes the finite difference measure noise.
            sciml_v = (; adaptive = false, dt = 1.0/240.0, solver = Huginn.ROCK2(),
                functional_inv = false, scalar = true)
            @testset "SciMLSensitivity adjoint with velocity loss vs finite differences" test_grad_finite_diff(
                ODINN.SciMLSensitivityAdjoint(); loss = LossV(),
                thres = [1e-4, 1e-10, 1e-4], sciml_v...)
            @testset "SciMLSensitivity adjoint with time-aggregated velocity loss vs finite differences" test_grad_finite_diff(
                ODINN.SciMLSensitivityAdjoint(); loss = LossAvgV(),
                aggregated_loss = :avgV, thres = [1e-4, 1e-10, 1e-4], sciml_v...)
            # The `:abs` component takes the norm of the averaged velocity, which is zero in the
            # cells without ice, where the derivative of a square root is infinite (0 × Inf = NaN)
            @testset "SciMLSensitivity adjoint with time-aggregated speed loss vs finite differences" test_grad_finite_diff(
                ODINN.SciMLSensitivityAdjoint(); loss = LossAvgV(component = :abs),
                aggregated_loss = :avgV, thres = [1e-4, 1e-10, 1e-4], sciml_v...)
            # H and V together: a binder bug only showed up with two loss terms combined.
            # `:time_span` and not the default `:uniform`: with `:uniform` all the weights are
            # one, so a wrong index into the weights would not show
            @testset "SciMLSensitivity adjoint with H and V losses and time_span weights vs finite differences" test_grad_finite_diff(
                ODINN.SciMLSensitivityAdjoint();
                loss = MultiLoss(
                    losses = (LossH(weighting = :time_span), LossV(weighting = :time_span)),
                    λs = (1.0, 1.0)),
                thres = [1e-4, 1e-10, 1e-4], sciml_v...)
            @testset "SciMLSensitivity adjoint with LossHV vs finite differences" test_grad_finite_diff(
                ODINN.SciMLSensitivityAdjoint(); loss = LossHV(),
                thres = [1e-4, 1e-10, 1e-4], sciml_v...)
            @testset "Continuous adjoint with LossHV vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); loss = LossHV(),
                thres = [1e-4, 1e-10, 1e-4], sciml_v...)
        end
    end

    if GROUP == "All" || GROUP == "Core5"
        @testset "Manual adjoint methods of SIA equation with hybrid D as target" begin
            @testset "Continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [1e-4, 2e-8, 2e-4], target = :D_hybrid)
            @testset "Continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = ContinuousVJP());
                thres = [2e-3, 3e-8, 2e-3], target = :D_hybrid)
        end
    end

    if GROUP == "All" || GROUP == "Core6"
        @testset "Adjoint method of SIA equation with pure D as target" begin
            @testset "Manual implementation of the continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [3e-2, 5e-5, 3e-2], target = :D)
            @testset "Manual implementation of the continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = ContinuousVJP());
                thres = [3e-2, 5e-5, 3e-2], target = :D)
            @testset "Manual implementation of the continuous adjoint with discrete VJP vs finite differences (loss V)" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [5e-3, 1e-6, 5e-3], target = :D, loss = LossV())
        end
    end
    if (GROUP == "All" && (!CI || !Sys.isapple())) || GROUP == "Core7"
        # Skip this test on macOS when running the "Full tests" CI because it is too slow and produces a timeout error (>6h)
        @testset "Adjoint method of SIA equation with pure D as target and custom NN" begin
            # @testset "Manual implementation of the continuous adjoint with discrete VJP and custom NN vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-3, 1e-7, 1e-3], target = :D, custom_NN = true)
            @testset "Manual implementation of the continuous adjoint with discrete VJP and custom NN vs finite differences (loss V)" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-4, 1e-7, 1e-4],
                target = :D, custom_NN = true, loss = LossV(),
                max_params = 25, mask_parameter_vector = true)
        end
    end

    if GROUP == "All" || GROUP == "Core8"
        @testset "Multi-objective function and regularization test" begin
            @testset "MultiLoss" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-3, 1e-8, 1e-3],
                loss = MultiLoss(losses = (LossH(),), λs = (0.4,)))
            @testset "MultiLoss SciMLSensitivity" test_grad_finite_diff(
                ODINN.SciMLSensitivityAdjoint(); thres = [5e-6, 1e-12, 5e-6],
                loss = MultiLoss(losses = (LossH(),), λs = (0.4,)))
            @testset "Just regularization" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-8, 1e-2],
                loss = MultiLoss(losses = (VelocityRegularization(),), λs = (1e2,)))
            @testset "Empirical and regularization" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [1e-4, 1e-8, 1e-4],
                loss = MultiLoss(losses = (LossH(), VelocityRegularization()), λs = (
                    1e-2, 2e-1)))
            @testset "Rheology regularization" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-8, 1e-8, 1e-8],
                functional_inv = false, scalar = false, loss = RheologyRegularization())
            # Checking the dhdt loss makes sense only with MB. With MB, FD needs fixed steps
            # like the other MB cases: with adaptive steps it measures step jitter.
            dhdt_fd = (functional_inv = false, scalar = true, loss = LossDhdt(),
                use_MB = true, aggregated_loss = :dhdt,
                adaptive = false, dt = 1.0/240.0, fd_delta = 1e-5)
            # DiscreteAdjoint steps back with explicit Euler, which is ~3% off on the MB
            # feedback ∂ṁ/∂H now that MB is continuous in the RHS
            @testset "Dhdt loss with discrete adjoint" test_grad_finite_diff(
                DiscreteAdjoint(VJP_method = DiscreteVJP()); thres_fd = 5e-2, dhdt_fd...)
            @testset "Dhdt loss with continuous adjoint" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres_fd = 5e-3, dhdt_fd...)
            if (!CI || !Sys.isapple())
                # The gradient computed with macOS within the CI is wrong
                # Despite a lot of effort we couldn't track the root cause, so we just deactivate that test
                @testset "AvgV loss with continuous adjoint" test_grad_finite_diff(
                    ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [
                        1e-3, 1e-8, 1e-3],
                    functional_inv = false, scalar = true, loss = LossAvgV(), aggregated_loss = :avgV)
            end
        end
        @testset "Joint inversion" begin
            @testset "Gridded A + IC" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-4, 1e-8, 1e-4],
                functional_inv = false, scalar = false,
                loss = MultiLoss(
                    losses = (LossH(), plain_ic_prior(2010.0)), λs = (
                        1.0, 1.0)),
                train_initial_conditions = true)
            @testset "IC regularization backward vs Zygote and FD" test_initial_thickness_regularization_backward()
            @testset "Sliding regularization backward vs Zygote and FD" test_sliding_regularization_backward()
        end
    end

    if GROUP == "All" || GROUP == "Core9"
        @testset "Classical inversions" begin
            @testset "Scalar inversion w/o MB" inversion_test(
                use_MB = false, multiprocessing = false, functional_inv = false)
            @testset "Gridded inversion w/o MB" inversion_test(
                use_MB = false, multiprocessing = false, functional_inv = false, scalar = false)
        end
        @testset "Functional inversions" begin
            @testset "Functional inversion w/o MB" inversion_test(use_MB = false, multiprocessing = false)
            # With MB in the RHS, abstol = 1e-3 leaves a loss floor of ~1e-4 where BFGS stops
            @testset "Functional inversion w/ MB" inversion_test(use_MB = true,
                multiprocessing = false, abstol = 1e-5,
                grad = ContinuousAdjoint(VJP_method = DiscreteVJP(regressorADBackend = DI.AutoZygote())))
            @testset "Functional inversion w/o MB w/ multiprocessing" inversion_test(
                use_MB = false, multiprocessing = true)
        end
    end

    if GROUP == "All" || GROUP == "Core10"
        @testset "Multiglacier inversion test" begin
            @testset "Continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                thres = [2e-4, 1e-8, 2e-4], multiglacier = true)
            # Two glaciers, one with the time window of the simulation and one with its own. The
            # gradient of the regularization is 3% of the one of the data, so it does not hide an error of the data term.
            @testset "Continuous adjoint with discrete VJP vs finite differences (initial condition, one glacier with its own time window)" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-3, 1e-8, 1e-3],
                multiglacier = true, train_initial_conditions = true, glacier_window = true,
                loss = MultiLoss(
                    losses = (LossH(), plain_ic_prior()), λs = (1.0, 1.0)))
            @testset "Discrete adjoint with discrete VJP vs finite differences (one glacier with its own time window)" test_grad_finite_diff(
                DiscreteAdjoint(VJP_method = DiscreteVJP()); thres = [5e-3, 1e-8, 5e-3],
                multiglacier = true, glacier_window = true)
            @testset "The start moves to the first survey when the initial condition is trained" test_align_start_in_inversion()
            @testset "The scale of the initial condition follows the glacier of each batch" test_ic_scale_in_batches()
            # With the automatic adjoint, the gradient of each glacier goes to its own place in θ
            # (`aggregate∇θ`), and it is not added to the one of the first glacier
            @testset "SciMLSensitivity adjoint, two glaciers, gridded C and initial condition" test_grad_finite_diff(
                SciMLSensitivityAdjoint();
                target = :C, functional_inv = false, scalar = false, maxC = 0.1, C₀ = 0.05,
                train_initial_conditions = true, multiglacier = true,
                loss = MultiLoss(
                    losses = (LossH(), plain_ic_prior(2010.0)), λs = (
                        1.0, 1.0)),
                A_range = (2e-18, 8e-18), adaptive = false, dt = 1.0/240.0, max_params = 25,
                thres = [1e-3, 1e-6, 2e-3])
        end
    end

    if GROUP == "All" || GROUP == "Core11"
        @testset "Save results" begin
            @testset "Single glacier" save_simulation_test!(multiglacier = false)
            @testset "Multiple glaciers" save_simulation_test!(multiglacier = true)
        end
    end

    if GROUP == "All" || GROUP == "Core12"
        # Mass balance evaluated in the ice flow right hand side, not applied as a jump —
        # the SciMLSensitivity case here was impossible before. Needs adaptive = false: with
        # an adaptive step the loss is discontinuous in θ, so a finite difference measures
        # step-acceptance jitter, not a derivative. See gradient_diagnostics.jl for how
        # `fd_delta`, `n_fd_components` and `thres_fd` below were chosen.
        fixed = (use_MB = true, A_range = (2e-18, 8e-18), n_fd_components = 2,
            adaptive = false, dt = 1.0/240.0, fd_delta = 1e-9, thres_fd = 5e-3)
        @testset "Mass balance as a continuous source term" begin
            @testset "Continuous adjoint vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); fixed...)
            @testset "SciMLSensitivity adjoint vs finite differences" test_grad_finite_diff(
                SciMLSensitivityAdjoint(); fixed...)
        end
    end

    if GROUP == "All" || GROUP == "Core13"
        # Gridded C, as in the sliding inversions. Only the automatic adjoint is tested: `LawC`
        # has no `p_VJP!`, so the manual adjoints give a zero gradient for θ.C. The test uses
        # ROCK4: with `RDPK3Sp35` the automatic adjoint gives a gradient with the wrong sign.
        fixed = (use_MB = true, A_range = (2e-18, 8e-18),
            adaptive = false, dt = 1.0/240.0, fd_delta = 1e-9, thres_fd = 5e-3)
        # `maxC` must be set: the default is for a Weertman law, about 15 orders of magnitude
        # too small for the Budd law used here, and the C gradient would be round off.
        # `LawC` starts at `C = maxC/2`, so `maxC = 0.1` gives as much sliding as deformation.
        @testset "Gridded C with the automatic adjoint" begin
            @testset "SciMLSensitivity adjoint vs finite differences" test_grad_finite_diff(
                SciMLSensitivityAdjoint();
                target = :C, functional_inv = false, scalar = false, maxC = 0.1, fixed...)
        end

        # Issue #409: initial condition with the automatic adjoint. Both θ.IC and θ.C are
        # checked, because a wrong fix can give the right θ.IC and a zero θ.C.
        @testset "Initial condition with the automatic adjoint" begin
            @testset "SciMLSensitivity adjoint vs finite differences" test_grad_finite_diff(
                SciMLSensitivityAdjoint();
                functional_inv = false, scalar = false, train_initial_conditions = true,
                loss = MultiLoss(
                    losses = (LossH(), plain_ic_prior(2010.0)),
                    λs = (1.0, 1.0)),
                fixed...)
        end

        # The setup of a transient inversion: gridded C inside (0, maxC) and H₀, H and V, and a
        # glacier on its own window that is not a whole number of mass balance steps (as when it
        # starts on a survey). Measured: ratio 3e-5, angle 3e-9 to 1e-8, relerr 9e-5 to 2e-4.
        @testset "Gridded C, initial condition, LossHV and own window with the automatic adjoint" test_grad_finite_diff(
            SciMLSensitivityAdjoint();
            target = :C, functional_inv = false, scalar = false, maxC = 0.1, C₀ = 0.05,
            train_initial_conditions = true, loss = LossHV(), glacier_window = true,
            window_offset = 1.0/36.0, adaptive = false, dt = 1.0/240.0, max_params = 25,
            thres = [1e-3, 1e-6, 1e-3])

        @testset "Free bed" test_free_bed()
        @testset "Value of each loss term, and the uncertainties" test_loss_terms()
        # Same setup with the bed free (unknowns H₀ and S₀, bed S₀ - H₀ inside the ice flow
        # and the velocity), and the surface constrained by two DEMs: one at the start of the
        # window and one inside it. Measured: ratio 2.3e-4, angle 1.5e-7, relerr 6e-4 to
        # 8.5e-4. The threshold of the relative error is 2e-3: it is set by the finite
        # difference of a fixed-step solve, and the bugs that this test is for give an O(1) error.
        @testset "Free bed, gridded C, LossHV and LossS with the automatic adjoint" test_grad_finite_diff(
            SciMLSensitivityAdjoint();
            target = :C, functional_inv = false, scalar = false, maxC = 0.1, C₀ = 0.05,
            train_initial_conditions = true, train_bed = true,
            loss = MultiLoss(losses = (LossHV(), LossS(), BedRegularization()),
                λs = (1.0, 1.0, 1.0)),
            glacier_window = true, window_offset = 1.0/36.0, adaptive = false,
            dt = 1.0/240.0, max_params = 25, thres = [1e-3, 1e-6, 2e-3])
        # The loss and the filter of the Vendr inversions: the velocity averaged over its
        # window (`LossAvgV`, differentiated through the velocity rrule that runs Enzyme) and the
        # `Zang1980` filter inside the bed `S₀ - H₀` that the ice flow differentiates
        @testset "Free bed, LossAvgV and the Zang1980 filter with the automatic adjoint" test_grad_finite_diff(
            SciMLSensitivityAdjoint();
            target = :C, functional_inv = false, scalar = false, maxC = 0.1, C₀ = 0.05,
            train_initial_conditions = true, train_bed = true,
            initial_condition_filter = :Zang1980, aggregated_loss = :avgV,
            loss = MultiLoss(losses = (LossH(), LossAvgV(), LossS(), BedRegularization())),
            adaptive = false, dt = 1.0/240.0, max_params = 25,
            thres = [1e-3, 1e-6, 2e-3])
    end

    # Manual, non-CI groups (GradTolGrid, GradTol, Core12NegativeControl, Core3MB,
    # Core3MBfd, Core12fd) live in gradient_diagnostics.jl, not here.
    include("gradient_diagnostics.jl")

    if GROUP == "All" || GROUP == "Aqua"
        @testset "Aqua" test_Aqua()
    end
end
