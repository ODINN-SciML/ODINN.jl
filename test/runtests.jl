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
        end
    end

    if GROUP == "All" || GROUP == "Core3"
        @testset "Manual adjoint methods of SIA equation with A as target" begin
            @testset "Discrete adjoint with discrete VJP vs finite differences" test_grad_finite_diff(
                DiscreteAdjoint(VJP_method = DiscreteVJP()); thres = [5e-3, 1e-8, 5e-3])
            @testset "Discrete adjoint with discrete VJP vs finite differences for scalar classical inversions" test_grad_finite_diff(
                DiscreteAdjoint(VJP_method = DiscreteVJP());
                functional_inv = false, thres = [5e-3, 1e-8, 5e-3])
            @testset "Discrete adjoint with discrete VJP vs finite differences (initial condition)" test_grad_finite_diff(
                DiscreteAdjoint(VJP_method = DiscreteVJP());
                thres = [5e-3, 5e-7, 5e-3], train_initial_conditions = true)
            @testset "Discrete adjoint with continuous VJP vs finite differences" test_grad_finite_diff(
                DiscreteAdjoint(VJP_method = ContinuousVJP()); thres = [1e-4, 1e-8, 1e-4])
            @testset "Continuous adjoint with discrete VJP vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-3, 1e-8, 1e-3])
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

            # The automatic adjoint had no velocity-loss coverage at all, and that is exactly
            # what let `batch_loss_iceflow_transient` read θ off the `InversionBinder` while
            # `solve` held the same binder as `p`: Zygote then drops the half of the gradient
            # that flows through the ODE, and the loss value stays exact so nothing errors.
            # `LossH` cannot catch it (it never uses θ), so the guard has to be a velocity
            # loss. Both cells below are 14.8x and 1.05x wrong respectively without the fix.
            # Fixed stepping and a pinned solver: with `solver = nothing` the harness picks a
            # different integrator per adjoint flavour, which confounds any comparison, and an
            # adaptive integrator makes the loss discontinuous in θ so the finite difference
            # measures step acceptance instead of a derivative.
            sciml_v = (; adaptive = false, dt = 1.0/240.0, solver = Huginn.ROCK2(),
                functional_inv = false, scalar = true)
            @testset "SciMLSensitivity adjoint with velocity loss vs finite differences" test_grad_finite_diff(
                ODINN.SciMLSensitivityAdjoint(); loss = LossV(),
                thres = [1e-4, 1e-10, 1e-4], sciml_v...)
            @testset "SciMLSensitivity adjoint with time-aggregated velocity loss vs finite differences" test_grad_finite_diff(
                ODINN.SciMLSensitivityAdjoint(); loss = LossAvgV(),
                aggregated_loss = :avgV, thres = [1e-4, 1e-10, 1e-4], sciml_v...)
            # H and V together: the second binder bug only showed up when two parts of the
            # loss were combined, so H and V passing on their own is not enough.
            @testset "SciMLSensitivity adjoint with H and V losses vs finite differences" test_grad_finite_diff(
                ODINN.SciMLSensitivityAdjoint();
                loss = MultiLoss(losses = (LossH(), LossV()), λs = (1.0, 1.0)),
                thres = [1e-4, 1e-10, 1e-4], sciml_v...)
            # With the default `:uniform` all the weights are one, so a wrong index into the
            # weights would not show
            @testset "SciMLSensitivity adjoint with time_span weights vs finite differences" test_grad_finite_diff(
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
                    losses = (LossH(), InitialThicknessRegularization(2010.0)), λs = (
                        1.0, 1.0)),
                train_initial_conditions = true)
            @testset "IC regularization backward vs Zygote and FD" test_initial_thickness_regularization_backward()
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
            @testset "Continuous adjoint with discrete VJP vs finite differences (initial condition)" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-3, 1e-8, 1e-3],
                multiglacier = true, train_initial_conditions = true)
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
        # Gridded C against the automatic adjoint. This cell of the matrix was empty: the
        # gridded classical inversion was only ever run with `ContinuousAdjoint`, and every
        # `SciMLSensitivityAdjoint` case used a scalar law, so nothing covered the combination
        # the sliding inversions actually use.
        #
        # What the gap cost: a campaign ran `RDPK3Sp35` with `SciMLSensitivityAdjoint` for
        # weeks. `InterpolatingAdjoint` is not stable with this ODE in backward mode, so the
        # gradient came back with the wrong sign — and with a perfectly healthy norm and no
        # NaNs, so nothing looked wrong. The harness picks ROCK4 here, which is the point.
        #
        # `LawC` defines no `p_VJP!`, so the manual adjoints would return a zero gradient for
        # θ.C rather than fail; only the automatic adjoint is meaningful for this target.
        fixed = (use_MB = true, A_range = (2e-18, 8e-18),
            adaptive = false, dt = 1.0/240.0, fd_delta = 1e-9, thres_fd = 5e-3)
        # `maxC` has to be set: the default only makes sense for a Weertman law, and against
        # the Budd law the glaciers are built with it is some fifteen orders too small. With it
        # the C gradient sits at round off (~1e-16) and the comparison below asserts nothing.
        # `LawC` starts at `C = maxC/2`, so 0.1 puts sliding on par with deformation at the
        # thicknesses involved, which is the regime the sliding inversions care about.
        @testset "Gridded C with the automatic adjoint" begin
            @testset "SciMLSensitivity adjoint vs finite differences" test_grad_finite_diff(
                SciMLSensitivityAdjoint();
                target = :C, functional_inv = false, scalar = false, maxC = 0.1, fixed...)
        end

        # Issue #409. `#366` added initial condition inversion and landed the manual adjoint
        # half; the SciMLSensitivity half was split out and never implemented. The problem was
        # being built outside the differentiated region, so `u0` stayed frozen at whatever θ it
        # was defined with and `θ.IC` carried no derivative through the solution at all —
        # measured `‖g.IC‖ = 1.7e-11` against `‖g.C‖ = 2.5e4`.
        #
        # Both blocks are checked. Checking only the one being fixed is how this was previously
        # got wrong: rebuilding `u0` from `container.θ` makes `θ.IC` correct and silently zeroes
        # `θ.C`, which a test of `θ.IC` alone would have passed.
        @testset "Initial condition with the automatic adjoint" begin
            @testset "SciMLSensitivity adjoint vs finite differences" test_grad_finite_diff(
                SciMLSensitivityAdjoint();
                functional_inv = false, scalar = false, train_initial_conditions = true,
                loss = MultiLoss(
                    losses = (LossH(), InitialThicknessRegularization(2010.0)),
                    λs = (1.0, 1.0)),
                fixed...)
        end
    end

    # Manual, non-CI groups (GradTolGrid, GradTol, Core12NegativeControl, Core3MB,
    # Core3MBfd, Core12fd) live in gradient_diagnostics.jl, not here.
    include("gradient_diagnostics.jl")

    if GROUP == "All" || GROUP == "Aqua"
        @testset "Aqua" test_Aqua()
    end
end
