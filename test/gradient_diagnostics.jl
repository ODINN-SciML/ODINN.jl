# Manual, non-CI groups for investigating the mass balance gradient by hand — step-size and
# tolerance sweeps that print rather than assert, plus one negative control. Run one with
# `GROUP=<name> julia test/runtests.jl`. None of these appear in .github/workflows/CI*.yml.

if GROUP == "GradTolGrid"
    # The scalar case cannot show direction bias: θ feeds a network with a single scalar
    # output, so every gradient is parallel to ∂A/∂θ whatever the solver does. A gridded
    # inversion gives θ one component per cell, where the direction is free to move.
    common = (; use_MB = true, A_range = (2e-18, 8e-18),
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
        A_range = (2e-18, 8e-18), solver = ROCK2(),
        abstol = 1e-6, return_grad = true)
    b = collect(base)
    println("  reference |g| = ", norm(b), "   (ROCK2, abstol 1e-6)")
    for atol in (1e-5, 1e-4, 1e-3, 1e-2)
        g = collect(test_grad_finite_diff(
            ContinuousAdjoint(VJP_method = DiscreteVJP()); use_MB = true,
            A_range = (2e-18, 8e-18), solver = ROCK2(),
            abstol = atol, return_grad = true))
        cosang = dot(g, b) / (norm(g) * norm(b))
        @printf("  abstol=%.0e  |g|=%.6e  rel|g| err=%.2e  angle=%.2e rad  relerr=%.2e\n",
            atol, norm(g), abs(norm(g) - norm(b)) / norm(b),
            acos(clamp(cosang, -1, 1)), norm(g .- b) / norm(b))
    end
end

if GROUP == "Core12NegativeControl"
    # Not part of any suite: it is expected to FAIL, and that is the point. Dropping the
    # elevation feedback must break the threshold the correct adjoints pass, otherwise
    # that threshold demonstrates nothing about the mass balance term.
    @testset "Negative control: elevation feedback removed (expected to fail)" test_grad_finite_diff(
        ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = NoVJP());
        use_MB = true, A_range = (2e-18, 8e-18),
        adaptive = false, dt = 1.0/240.0, fd_delta = 1e-9, thres_fd = 5e-3)
end

if GROUP == "Core3MB"
    # The mass balance gradient cases of Core3 on their own, with the same settings:
    # Core3 takes over an hour and most of it is unrelated to them.
    mb_fd = (adaptive = false, dt = 1.0/240.0, fd_delta = 1e-5, thres_fd = 5e-3)
    @testset "MB gradient cases" begin
        @testset "Enzyme MB VJP" test_grad_finite_diff(
            ContinuousAdjoint(
                VJP_method = DiscreteVJP(regressorADBackend = DI.AutoZygote()),
                MB_VJP = ODINN.EnzymeVJP());
            thres = [3e-3, 1e-8, 3e-3], use_MB = true, mb_fd...)
        @testset "discrete MB VJP" test_grad_finite_diff(
            ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = DiscreteVJP());
            thres = [3e-3, 1e-8, 3e-3], use_MB = true, mb_fd...)
        @testset "discrete MB VJP w/ temperature bias" test_grad_finite_diff(
            ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = DiscreteVJP());
            thres = [3e-3, 1e-8, 3e-3], use_MB = true, temp_bias = 1.0, mb_fd...)
        @testset "Enzyme MB VJP w/ temperature bias" test_grad_finite_diff(
            ContinuousAdjoint(
                VJP_method = DiscreteVJP(regressorADBackend = DI.AutoZygote()),
                MB_VJP = ODINN.EnzymeVJP());
            thres = [3e-3, 1e-8, 3e-3], use_MB = true, temp_bias = 1.0, mb_fd...)
        @testset "discrete MB VJP w/ calibrated MB" test_grad_finite_diff(
            ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = DiscreteVJP());
            thres = [2e-3, 1e-10, 2e-3], use_MB = true, calibrate_MB = true, mb_fd...)
    end
end

if GROUP == "Core3MBfd"
    # Diagnostic. These cases shrink A so the gradient is mass balance dominated, which
    # also makes it ~1e-3 where Core12's is ~1: a thousand times smaller, so the step that
    # suits Core12 sits deep in the cancellation regime here. Sweep it and see whether the
    # difference converges onto the adjoint, as it did for Core12.
    for δ in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
        @printf("\n  fd_delta = %.0e\n", δ)
        test_grad_finite_diff(
            ContinuousAdjoint(VJP_method = DiscreteVJP(), MB_VJP = DiscreteVJP());
            use_MB = true, adaptive = false, dt = 1.0/240.0,
            fd_delta = δ, thres_fd = 1e9, thres = [1e9, 1e9, 1e9])
    end
end

if GROUP == "Core12fd"
    # Diagnostic, never part of a suite. A fixed-step central difference is least
    # accurate on the smallest gradient component, which is where cancellation bites.
    # Sweeping the step tells the two explanations apart: a noisy finite difference
    # scatters around the adjoint, a wrong adjoint has the difference converge somewhere
    # else. `thres_fd` is wide open so nothing here can fail.
    #
    # This is what backs Core12's `fd_delta = 1e-9`: on the smallest gradient component,
    # 1e-13 returns the wrong sign, 1e-11 is still 1.0e-1 off, and 1e-9 lands within 3e-4.
    # Anything below ~1e-10 measures cancellation, not a derivative.
    for δ in (1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8)
        @printf("\n  fd_delta = %.0e\n", δ)
        test_grad_finite_diff(
            ContinuousAdjoint(VJP_method = DiscreteVJP());
            use_MB = true, A_range = (2e-18, 8e-18),
            adaptive = false, dt = 1.0/240.0, fd_delta = δ, thres_fd = 1e9)
    end
end
