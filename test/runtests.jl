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
            @testset "Continuous adjoint with discrete VJP vs finite differences w/ Enzyme MB VJP" test_grad_finite_diff(
                ContinuousAdjoint(
                    VJP_method = DiscreteVJP(regressorADBackend = DI.AutoZygote()),
                    MB_VJP = ODINN.EnzymeVJP());
                thres = [3e-3, 1e-8, 3e-3],
                use_MB = true) # This test uses Zygote for the differentiation of the laws because Mooncake has to store modules inside the VJPsPrepLaw struct which is not compatible with Enzyme.make_zero
            @testset "Continuous adjoint with discrete VJP vs finite differences w/ discrete MB VJP" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = DiscreteVJP());
                thres = [3e-3, 1e-8, 3e-3], use_MB = true)
            # A nonzero temp_bias moves the PDD/snow clamp thresholds, which the MB VJPs
            # must pick up from the MB model rather than assume away.
            @testset "Continuous adjoint w/ discrete MB VJP and temperature bias" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = DiscreteVJP());
                thres = [3e-3, 1e-8, 3e-3], use_MB = true, temp_bias = 1.0)
            @testset "Continuous adjoint w/ Enzyme MB VJP and temperature bias" test_grad_finite_diff(
                ContinuousAdjoint(
                    VJP_method = DiscreteVJP(regressorADBackend = DI.AutoZygote()),
                    MB_VJP = ODINN.EnzymeVJP());
                thres = [3e-3, 1e-8, 3e-3], use_MB = true, temp_bias = 1.0)
            # Calibration turns mass_balance into one model per glacier, so this covers the
            # vector of MB models flowing through the VJPs.
            @testset "Continuous adjoint w/ discrete MB VJP and calibrated MB" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = DiscreteVJP());
                thres = [2e-3, 1e-10, 2e-3], use_MB = true, calibrate_MB = true)
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
                thres = [1e-2, 1e-5, 1e-2], loss = LossV())
            @testset "Continuous adjoint with discrete VJP vs finite differences (Log)" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [1e-2, 1e-5, 1e-2],
                loss = LossV(loss = LogSum(), component = :abs))
            # @testset "Continuous adjoint with continuous VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ContinuousVJP()); thres = [2e-2, 1e-5, 2e-2], loss=LossV())
            # @testset "Continuous adjoint with Enzyme VJP vs finite differences" test_grad_finite_diff(ContinuousAdjoint(VJP_method = ODINN.EnzymeVJP()); thres = [2e-4, 1e-8, 1e-3], loss=LossV())
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
            @testset "Dhdt loss with discrete adjoint" test_grad_finite_diff( # Checking the dhdt loss makes sense only with MB
                DiscreteAdjoint(VJP_method = DiscreteVJP()); thres = [5e-3, 1e-8, 5e-3],
                functional_inv = false, scalar = true, loss = LossDhdt(), use_MB = true, aggregated_loss = :dhdt)
            @testset "Dhdt loss with continuous adjoint" test_grad_finite_diff( # Checking the dhdt loss makes sense only with MB
                ContinuousAdjoint(VJP_method = DiscreteVJP()); thres = [5e-3, 1e-8, 5e-3],
                functional_inv = false, scalar = true, loss = LossDhdt(), use_MB = true, aggregated_loss = :dhdt)
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
            @testset "Functional inversion w/ MB" inversion_test(use_MB = true,
                multiprocessing = false,
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
        # Mass balance evaluated in the ice flow right hand side instead of applied as a jump.
        # The SciMLSensitivity case is the one that was impossible before: differentiating the
        # periodic callback was unsupported, so MB and the automatic adjoint were exclusive.
        #
        # These use a fixed step and a fixed difference. With an adaptive integrator the loss
        # is a discontinuous function of θ — an arbitrarily small change flips which steps are
        # accepted — so a finite difference measures that jitter instead of a derivative, no
        # matter how the step is chosen.
        #
        # `fd_delta` matters more than it looks. Sweeping it (GROUP = "Core12fd") shows the
        # difference converging onto the adjoint as the step grows: on the smallest gradient
        # component, 1e-13 returns the wrong sign, 1e-11 is still 1.0e-1 off, and 1e-9 lands
        # within 3e-4. Anything below ~1e-10 measures cancellation, not a derivative. The
        # teeth testset below is what keeps the threshold honest.
        fixed = (use_MB = true, MB_scheme = :continuous, A_range = (2e-18, 8e-18),
            adaptive = false, dt = 1.0/240.0, fd_delta = 1e-9, thres_fd = 5e-2)
        @testset "Mass balance as a continuous source term" begin
            @testset "Continuous adjoint vs finite differences" test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); fixed...)
            @testset "SciMLSensitivity adjoint vs finite differences" test_grad_finite_diff(
                SciMLSensitivityAdjoint(); fixed...)
        end
    end

    if GROUP == "GradTolGrid"
        # The scalar case cannot show direction bias: θ feeds a network with a single scalar
        # output, so every gradient is parallel to ∂A/∂θ whatever the solver does. A gridded
        # inversion gives θ one component per cell, where the direction is free to move.
        common = (; use_MB = true, MB_scheme = :continuous, A_range = (2e-18, 8e-18),
            solver = ROCK2(), functional_inv = false, scalar = false, return_grad = true)
        b = collect(test_grad_finite_diff(
            ContinuousAdjoint(VJP_method = DiscreteVJP()); abstol = 1e-6, common...))
        @printf("  reference |g| = %.6e over %d components (ROCK2, abstol 1e-6)\n",
            norm(b), length(b))
        for atol in (1e-5, 1e-4, 1e-3, 1e-2)
            g = collect(test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); abstol = atol, common...))
            cosang = dot(g, b) / (norm(g) * norm(b))
            @printf("  abstol=%.0e  |g|=%.6e  rel|g|=%.2e  angle=%.3e rad (%.3f deg)  relerr=%.2e\n",
                atol, norm(g), abs(norm(g) - norm(b)) / norm(b),
                acos(clamp(cosang, -1, 1)), rad2deg(acos(clamp(cosang, -1, 1))),
                norm(g .- b) / norm(b))
        end
    end

    if GROUP == "GradTol"
        # Diagnostic. Everything else measures H accuracy; for an inversion what matters is
        # whether a loose tolerance biases the GRADIENT. Direction error is reported apart
        # from magnitude because an optimiser follows the direction.
        base = test_grad_finite_diff(
            ContinuousAdjoint(VJP_method = DiscreteVJP()); use_MB = true,
            MB_scheme = :continuous, A_range = (2e-18, 8e-18), solver = ROCK2(),
            abstol = 1e-6, return_grad = true)
        b = collect(base)
        println("  reference |g| = ", norm(b), "   (ROCK2, abstol 1e-6)")
        for atol in (1e-5, 1e-4, 1e-3, 1e-2)
            g = collect(test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP()); use_MB = true,
                MB_scheme = :continuous, A_range = (2e-18, 8e-18), solver = ROCK2(),
                abstol = atol, return_grad = true))
            cosang = dot(g, b) / (norm(g) * norm(b))
            @printf("  abstol=%.0e  |g|=%.6e  rel|g| err=%.2e  angle=%.2e rad  relerr=%.2e\n",
                atol, norm(g), abs(norm(g) - norm(b)) / norm(b),
                acos(clamp(cosang, -1, 1)), norm(g .- b) / norm(b))
        end
    end

    if GROUP == "Core12teeth"
        # Not part of any suite: it is expected to FAIL, and that is the point. Dropping the
        # elevation feedback must break the threshold the correct adjoints pass, otherwise
        # that threshold demonstrates nothing about the mass balance term.
        @testset "Teeth: elevation feedback removed (expected to fail)" test_grad_finite_diff(
            ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = NoVJP());
            use_MB = true, MB_scheme = :continuous, A_range = (2e-18, 8e-18),
            adaptive = false, dt = 1.0/240.0, fd_delta = 1e-9, thres_fd = 5e-2)
    end

    if GROUP == "Core12fd"
        # Diagnostic, never part of a suite. A fixed-step central difference is least
        # accurate on the smallest gradient component, which is where cancellation bites.
        # Sweeping the step tells the two explanations apart: a noisy finite difference
        # scatters around the adjoint, a wrong adjoint has the difference converge somewhere
        # else. `thres_fd` is wide open so nothing here can fail.
        for δ in (1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8)
            @printf("\n  fd_delta = %.0e\n", δ)
            test_grad_finite_diff(
                ContinuousAdjoint(VJP_method = DiscreteVJP());
                use_MB = true, MB_scheme = :continuous, A_range = (2e-18, 8e-18),
                adaptive = false, dt = 1.0/240.0, fd_delta = δ, thres_fd = 1e9)
        end
    end

    if GROUP == "All" || GROUP == "Aqua"
        @testset "Aqua" test_Aqua()
    end
end
