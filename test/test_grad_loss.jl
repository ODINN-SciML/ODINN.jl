using Distributed: map

"""
    jet_opt_clean(f, tt, modules) -> Bool

Whether JET's optimization analysis of `f` on argument types `tt` reports nothing.

Returns `false` when JET itself raises. JET 0.9 is the newest release usable on Julia 1.11
and, on entry points as deeply nested as the gradient ones, it can throw from its own
constant propagation rather than reporting. That is not information about `f`, and it is why
these checks call JET directly instead of through `@test_opt`: the macro records the throw as
an error, which a `broken` marker cannot absorb.
"""
function jet_opt_clean(f, tt, modules)
    try
        return isempty(JET.get_reports(JET.report_opt(f, tt; target_modules = modules)))
    catch err
        err isa TypeError || rethrow()
        return false
    end
end

"""
    test_grad_finite_diff(
        adjointFlavor::ADJ;
        thres = [0., 0., 0.],
        target = :A,
        finite_difference_order = 3,
        loss = LossH(),
        train_initial_conditions = false,
        multiglacier = false,
        glacier_window = false,
        single_survey = false,
        use_MB = false,
        temp_bias = 0.0,
        calibrate_MB = false,
        abstol = 1e-6,
        solver = nothing,
        A_range = nothing,
        adaptive = true,
        dt = 1.0/120.0,
        fd_delta = nothing,
        thres_fd = 5e-2,
        n_fd_components = 4,
        return_grad = false,
        functional_inv = true,
        scalar = true,
        custom_NN = false,
        max_params = 60,
        mask_parameter_vector = false,
        aggregated_loss = nothing,
    ) where {ADJ<:AbstractAdjointMethod}

Test and validate gradient consistency between adjoint-based automatic differentiation
and finite-difference approximations.

This function sets up a controlled glaciological simulation with configurable physical
and neural-network components, computes model gradients using both the specified adjoint
method and finite-difference schemes, and compares them using diagnostic metrics.

# Arguments

  - `adjointFlavor::ADJ`: The adjoint computation method to test, e.g. `ODINN.SciMLSensitivityAdjoint` or other `AbstractAdjointMethod` subtypes.
  - `thres::Vector{<:Real}`: Three-element vector of numerical thresholds for
    `(ratio, angle, relative error)` comparison between adjoint-based and finite-difference gradients.
  - `target::Symbol`: Model target for training/testing (`:A`, `:D`, or `:D_hybrid`), determining which physical law is parameterized by the neural network.
  - `finite_difference_order::Int`: Order of accuracy for central finite differences.
  - `loss`: Loss function to evaluate, such as `LossH()` (height-based) or `LossV()` (velocity-based).
  - `train_initial_conditions::Bool`: Whether to include glacier initial conditions as trainable parameters.
  - `multiglacier::Bool`: Whether to run the test on multiple glaciers.
  - `glacier_window::Bool`: If `true`, the last glacier has its own time window, which starts on its third observation. The observations before are outside its window.
  - `window_offset`: With `glacier_window`, the window starts this many years after the third observation. A value that is not a multiple of the mass balance step makes the window a non-whole number of steps, as when a run starts on a survey: the last mass balance window ends after the end of the run.
  - `single_survey::Bool`: If `true`, the first glacier keeps only one thickness observation, in the middle of the time window.
  - `C₀`: Initial `glacier.C` of all the glaciers, set after the ground truth is generated. With `nothing`, `C = 0`, and a gridded `C` starts in the flat part of its `tanh` map, where its gradient is tiny. Use a value inside `(0, maxC)` to test `C`.
  - `train_bed::Bool`: If `true`, the bed is free (`InitialCondition` with `bed = :free`, which needs `train_initial_conditions`), and each glacier gets two DEMs for `LossS`: one at the start of its time window, and one in the middle of `tspan`, 2 m higher, with one ice cell without value. Otherwise the initial condition uses `bed = :fixed`.
  - `initial_condition_filter::Symbol`: Filter that keeps the initial thickness non-negative (`:softplus`, `:Zang1980` or `:identity`), see `evaluate_H₀`.
  - `use_MB::Bool`: Whether to include a mass balance model (MB) during training/testing.
  - `temp_bias`: Temperature bias of the tested `TImodel1`, in °C. A nonzero value shifts
    where the PDD and snow clamps activate, so it exercises that dependency in the MB VJP.
  - `calibrate_MB::Bool`: Whether to calibrate the mass balance model against the geodetic
    observations. Off by default so the `TImodel1` built below is the one actually tested;
    turn it on to check the adjoint against a per-glacier calibrated vector of MB models.
  - `abstol`: Absolute tolerance of the ODE solver, in metres of ice. It only matters with `adaptive = true`, and it is tightened for runs longer than 5 years (see `Huginn.effective_abstol`).
  - `solver`: ODE solver. By default `ROCK4()` with `SciMLSensitivityAdjoint`, because the backward solve of `InterpolatingAdjoint` is not stable with `RDPK3Sp35` on our ODE, and `RDPK3Sp35()` otherwise. The spectral radius is supplied to `ROCK2` and `ROCK4` (`supply_eigen_est = true`), since their own estimate breaks finite differences.
  - `A_range`: `(minA, maxA)`, the range of the rheology `A`. If `nothing`, it depends on the case: `(2e-18, 8e-18)` by default and for the `:dhdt` and `:avgV` losses, and `(1e-21, 2e-21)` with `use_MB`, so that the gradient is dominated by the mass balance.
  - `adaptive::Bool`: Whether the solver picks its own steps. Fixed-step finite differences (`fd_delta`) need `false`, because with adaptive steps the loss is discontinuous in `θ`.
  - `dt`: Fixed step in years, used when `adaptive = false`.
  - `fd_delta`: If not `nothing`, the adjoint is compared with a central finite difference at this fixed step, instead of using `FiniteDifferences.jl` with its step-size search. It is computed for the first `n_fd_components` components of `θ`, and it requires `adaptive = false`. `thres` is not used in this case.
  - `thres_fd`: Threshold on the relative error `|fd - adjoint| / |fd|` of each component in the fixed-step comparison.
  - `n_fd_components`: Number of components of `θ` checked in the fixed-step comparison.
  - `return_grad::Bool`: Return the gradient computed by the adjoint and skip the finite-difference comparison.
  - `functional_inv::Bool`: Whether to test functional inversions or classical inversions.
  - `scalar::Bool`: Whether the rheology `A` is a single scalar per glacier (`true`) or a gridded field (`false`).
  - `custom_NN::Bool`: Whether to use a custom-defined neural network architecture for testing or a simple default small network. If the custom neural network is used, the glacier grid and the number of points in the VJP interpolation are reduced to spare computation time and memory.
  - `max_params::Int`: Maximum number of parameters for finite-difference testing; if exceeded, a random subset is tested to reduce computational cost.
  - `mask_parameter_vector::Bool`: Whether to apply a mask to the parameter vector `θ` before evaluating finite-difference gradients. If `false`, the
    mask based on `max_params` is just applied to the initial conditions, not to parameters of the regressor.
  - `aggregated_loss`: `nothing` (default), `:dhdt` or `:avgV`. With `:dhdt` or `:avgV` the loss compares the elevation change or the average velocity over the whole period, instead of the state at each time step. It needs `LossDhdt()` or `LossAvgV()` respectively. The `:dhdt` case also uses the period 2010–2015 instead of the 1980–2019 of `use_MB`, because over the longer one the melt empties the mask and the gradient vanishes, and a stronger melt, so that `dhdt` is negative without melting the glacier out.
"""
function test_grad_finite_diff(
        adjointFlavor::ADJ;
        thres = [0.0, 0.0, 0.0],
        target = :A,
        finite_difference_order = 3,
        loss = LossH(),
        train_initial_conditions = false,
        multiglacier = false,
        glacier_window = false,
        window_offset = 0.0,
        single_survey = false,
        C₀ = nothing,
        train_bed = false,
        initial_condition_filter = :softplus,
        use_MB = false,
        temp_bias = 0.0,
        calibrate_MB = false,
        abstol = 1e-6,
        solver = nothing,
        A_range = nothing,
        # The default `maxC` is for a Weertman law. It is far too small for the Budd law used
        # here, so the C gradient is round off. Tests with target `:C` must set it.
        maxC = Sleipnir.PhysicalParameters().maxC,
        adaptive = true,
        dt = 1.0/120.0,
        fd_delta = nothing,
        thres_fd = 5e-2,
        n_fd_components = 4,
        return_grad = false,
        # Return the built `(simulation, θ)` without running the gradient, to test one part of
        # the adjoint on the same setup.
        return_setup = false,
        # Override the sensealg of the automatic adjoint, to compare VJP backends. Velocity
        # losses were 14.8x wrong through the automatic path (thickness losses were exact),
        # because their seeds come from an rrule that runs Enzyme inside the Zygote pullback.
        sensealg = nothing,
        functional_inv = true,
        scalar = true,
        custom_NN = false,
        max_params = 60,
        mask_parameter_vector = false,
        aggregated_loss = nothing
) where {ADJ <: AbstractAdjointMethod}
    if !functional_inv
        @assert target in (:A, :C) "When testing classical inversion, only targets A and C are supported"
    end

    print("> Testing target $(target) with $(adjointFlavor) and $(Base.typename(typeof(loss)).name)")
    println(use_MB ? " and with MB" : "")

    # Determine if we are working with a velocity loss
    velocityLoss = ODINN.loss_uses_velocity(loss)

    thres_ratio = thres[1]
    thres_angle = thres[2]
    thres_relerr = thres[3]

    rgi_ids = @match (velocityLoss, multiglacier, aggregated_loss) begin
        (true, false, nothing) => ["RGI60-11.03646"]
        (false, false, nothing) => ["RGI60-11.03638"]
        (false, true, nothing) => ["RGI60-11.03638", "RGI60-11.01450"]
        (false, false, :dhdt) => ["RGI60-11.03638"]
        (true, false, :avgV) => ["RGI60-11.03646"]
    end

    rgi_paths = get_rgi_paths()

    working_dir = joinpath(ODINN.root_dir, "test/data")

    δt = 1/12
    # Short window: over 1980-2019 the melt empties the mask and ∂dhdt/∂A vanishes.
    tspan = if aggregated_loss == :dhdt
        (2010.0, 2015.0)
    elseif use_MB
        (1980.0, 2019.0)
    else
        (2010.0, 2012.0)
    end

    useSciMLSenseAlg = isa(adjointFlavor, ODINN.SciMLSensitivityAdjoint)
    sensealg_override = sensealg
    if useSciMLSenseAlg
        optim_autoAD = Optimization.AutoEnzyme()
        sensealg = isnothing(sensealg_override) ?
                   InterpolatingAdjoint(autojacvec = SciMLSensitivity.EnzymeVJP()) :
                   sensealg_override
    else
        optim_autoAD = ODINN.NoAD()
        sensealg = isnothing(sensealg_override) ? SciMLSensitivity.ZygoteAdjoint() :
                   sensealg_override
    end

    minA, maxA = if !isnothing(A_range)
        A_range
    elseif aggregated_loss == :dhdt || aggregated_loss == :avgV
        (2e-18, 8e-18)
    else
        # When MB is being tested, reduce the impact of creeping so that the gradient is dominated by the MB contribution
        use_MB ? (1e-21, 2e-21) : (2e-18, 8e-18)
    end

    params = Parameters(
        simulation = SimulationParameters(
            working_dir = working_dir,
            use_MB = use_MB,
            use_velocities = true,
            tspan = tspan,
            step_MB = δt,
            multiprocessing = false,
            workers = 1,
            test_mode = true,
            calibrate_MB = calibrate_MB,
            rgi_paths = rgi_paths,
            gridScalingFactor = custom_NN ? 8 : 4,
            f_surface_velocity_factor = 0.8
        ),
        hyper = Hyperparameters(
            batch_size = length(rgi_ids), # We set batch size equals all datasize so we test gradient
            epochs = 100,
            optimizer = ODINN.Adam(0.005)
        ),
        physical = PhysicalParameters(
            minA = minA,
            maxA = maxA,
            maxC = maxC
        ),
        UDE = UDEparameters(
            sensealg = sensealg,
            optim_autoAD = optim_autoAD,
            grad = adjointFlavor,
            optimization_method = "AD+AD",
            empirical_loss_function = loss,
            # Only the manual adjoints use `UDE.target`. A C inversion still carries
            # `SIA2D_A_target`, so it stays `:A` to pass the `Inversion` constructor check.
            target = target == :C ? :A : target,
            initial_condition_filter = initial_condition_filter
        ),
        solver = Huginn.SolverParameters(
            step = δt,
            abstol = abstol,
            adaptive = adaptive,
            dt = dt,
            progress = true,
            # ROCK's own estimate shifts the scheme between neighbouring θ, breaking FD
            supply_eigen_est = true,
            solver = if !isnothing(solver)
                solver
            else
                useSciMLSenseAlg ? ROCK4() : RDPK3Sp35() # Use another solver when using SciMLSensitivity because `InterpolatingAdjoint` is not stable with our ODE in backward mode
            end
        )
    )

    # We retrieve some glaciers for the simulation
    # Time snapshots for transient inversion
    tstops = collect(tspan[1]:δt:tspan[2])

    kwargs = velocityLoss ?
             (;
        velocityDatacubes = Dict(
        rgi_ids[1] => Sleipnir.fake_multi_datacube()
    )
    ) : NamedTuple()
    ground_truth_A_law = scalar ? ConstantA(2.21e-18) : CuffeyPaterson(scalar = scalar)
    model = Model(
        iceflow = SIA2Dmodel(params; A = ground_truth_A_law),
        mass_balance = TImodel1(params; DDF = 6.0/1000.0, prcp_fac = 1.2)
    )
    glaciers = initialize_glaciers(rgi_ids, params; kwargs...)
    if !functional_inv
        for i in 1:length(glaciers)
            glaciers[i].A = 4e-18
        end
    end

    # Generate ground truth based on the loss that will be used hereafter
    store = if aggregated_loss == :dhdt
        (:H, :dhdt)
    elseif aggregated_loss == :avgV
        (:H, :avgV)
    else
        ODINN.loss_uses_velocity(loss) ? (:H, :V) : (:H,)
    end
    glaciers = generate_ground_truth(glaciers, params, model, tstops; store = store)
    if glacier_window
        params.simulation.glacier_tspans[glaciers[end].rgi_id] = (
            tstops[3] + window_offset, tspan[2])
    end
    if single_survey
        # Keep only one thickness survey, in the middle of the time window
        td = glaciers[1].thicknessData
        k = cld(length(td.t), 2)
        glaciers[1] = Sleipnir.Glacier2D(glaciers[1];
            thicknessData = Sleipnir.ThicknessData(t = [td.t[k]], H = [td.H[k]],
                σ = isnothing(td.σ) ? nothing : [td.σ[k]]))
    end

    isnothing(C₀) || foreach(g -> g.C = C₀, glaciers)

    if train_bed
        glaciers = map(glaciers) do glacier
            S₂ = glacier.S .+ 2.0
            σ₂ = ones(size(S₂))
            k = findfirst(>(0), glacier.H₀)
            S₂[k] = σ₂[k] = NaN
            surfaceData = Sleipnir.SurfaceData(
                t = [glacier_tspan(params, glacier)[1], tstops[cld(end, 2)]],
                S = [copy(glacier.S), S₂], σ = [zero(glacier.S), σ₂])
            Sleipnir.Glacier2D(glacier; surfaceData = surfaceData)
        end
    end

    # Neural network model
    if functional_inv
        if custom_NN
            architecture = Lux.Chain(
                Lux.WrappedFunction(x -> LuxFunction(
                    v -> ODINN.normalize(v; lims = ([0.0, 0.0], [200.0, 0.6])), x)),
                Lux.Dense(2, 5, x -> Lux.gelu.(x)),
                Lux.Dense(5, 10, x -> Lux.gelu.(x)),
                Lux.Dense(10, 5, x -> Lux.gelu.(x)),
                Lux.Dense(5, 1, sigmoid),
                Lux.WrappedFunction(x -> LuxFunction(v -> v*1e2, x))
            )
            nn_model = NeuralNetwork(params; architecture = architecture)
        else
            nn_model = NeuralNetwork(params)
        end
    end

    @assert !train_bed || train_initial_conditions "The bed is free only with the initial condition"
    ic = train_initial_conditions ?
         InitialCondition(params, glaciers, :Farinotti19; bed = train_bed ? :free :
                                                                :fixed) :
         nothing
    trainable_model = if functional_inv
        nn_model
    elseif scalar
        GlacierWideInv(params, glaciers, target)
    else
        GriddedInv(params, glaciers, target)
    end

    # Define regressors for each test
    regressors = @match (target, train_initial_conditions) begin
        (:A, false) => (; A = trainable_model)
        (:A, true) => (; A = trainable_model, IC = ic)
        (:C, false) => (; C = trainable_model)
        (:C, true) => (; C = trainable_model, IC = ic)
        (:D_hybrid, false) => (; Y = trainable_model)
        (:D_hybrid, true) => (; Y = trainable_model, IC = ic)
        (:D, false) => (; U = trainable_model)
        (:D, true) => (; U = trainable_model, IC = ic)
    end

    law = @match (target, functional_inv) begin
        (:A, true) => LawA(trainable_model, params; scalar = scalar)
        (:A, false) => LawA(params; scalar = scalar)
        # Classical only: `LawC` has no neural network version yet
        (:C, false) => LawC(params; scalar = scalar)
        (:D_hybrid, true) => LawY(trainable_model, params)
        (:D, true) => LawU(trainable_model, params)
    end

    mass_balance = if aggregated_loss==:dhdt
        # Intensify melting to make dhdt negative, without melting the glacier out.
        TImodel1(params; DDF = 9.0/1000.0, prcp_fac = 0.8, temp_bias = temp_bias)
    else
        TImodel1(params; DDF = 6.0/1000.0, prcp_fac = 1.2, temp_bias = temp_bias)
    end
    model = @match target begin
        :A => Model(
            iceflow = SIA2Dmodel(params; A = law),
            mass_balance = mass_balance,
            regressors = regressors
        )
        :C => Model(
            iceflow = SIA2Dmodel(params; C = law),
            mass_balance = mass_balance,
            regressors = regressors
        )
        :D_hybrid => Model(
            iceflow = SIA2Dmodel(params; Y = law),
            mass_balance = mass_balance,
            regressors = regressors
        )
        :D => Model(
            iceflow = SIA2Dmodel(params; U = law),
            mass_balance = mass_balance,
            regressors = regressors,
            target = SIA2D_D_target(
                interpolation = :Linear,
                n_interp_half = custom_NN ? 50 : 200
            )
        )
    end

    # We create an ODINN prediction
    simulation = Inversion(model, glaciers, params)
    θ = simulation.model.trainable_components.θ
    n_params = length(θ)

    if (:A in keys(θ)) && (Symbol("1") in keys(θ.A)) # Classical inversion
        if length(θ.A) != length(glaciers) # Gridded inversion
            for i in 1:length(glaciers)
                # Perturb the parameterization so that the value of the regularization is not zero (constant matrix is in the null space of the Tikhonov regularization)
                θ.A[Symbol("$(i)")] .*= 0.5 .+ rand(Float64, size(θ.A[Symbol("$(i)")]))
            end
        end
    end

    return_setup && return (simulation, θ)

    loss_iceflow_grad!(dθ, _θ, _simulation) =
        if useSciMLSenseAlg
            ret = ODINN.grad_loss_iceflow!(_θ, simulation, map)
            @assert !any(isnan, ret) "Gradient computed with SciML contains NaNs. Try to run the code again if you just started the REPL. Gradient is $(ret)"
            dθ .= ret
        else
            SIA2D_grad!(dθ, _θ, _simulation)
        end

    function f(x, simulation)
        simulation.model.trainable_components.θ = x
        return ODINN.loss_iceflow_transient(x, simulation, map)
    end

    dθ = zero(θ)
    if !isa(adjointFlavor, ODINN.SciMLSensitivityAdjoint)
        loss_iceflow_grad!(dθ, θ, simulation)
    else
        loss_iceflow_grad!(dθ, θ, simulation)
    end
    jet_modules = (Sleipnir, Muninn, Huginn, ODINN)
    @test_broken jet_opt_clean(
        loss_iceflow_grad!, Tuple{typeof(dθ), typeof(θ), typeof(simulation)}, jet_modules)
    @test_broken jet_opt_clean(ODINN.loss_iceflow_transient,
        Tuple{typeof(θ), typeof(simulation), typeof(map)}, jet_modules)

    return_grad && return dθ

    if !isnothing(fd_delta)
        # Central difference at a fixed step. The adaptive step of FiniteDifferences is
        # unusable here: with an adaptive integrator the loss is discontinuous in θ, so it
        # would measure step-acceptance jitter. This path requires `adaptive = false`.
        @assert !adaptive "A fixed step finite difference is only meaningful with adaptive = false."
        θ0 = deepcopy(θ)

        # A gradient at round off passes the relative comparisons below, because the finite
        # difference is zero too. So require it to be resolvable against the loss scale (for
        # example, a gridded C with the default `maxC` and a Budd law is not).
        l0 = f(θ0, simulation)
        @test maximum(abs, collect(dθ)) >
              sqrt(eps(Sleipnir.Float)) * max(one(Sleipnir.Float), abs(l0))

        # Rank by magnitude rather than taking the first components. For a scalar parameter the
        # two are the same, but for a gridded one the leading components are ice-free corner
        # cells where both sides are exactly zero and the comparison asserts nothing.
        fd_idx = sortperm(abs.(vec(collect(dθ))); rev = true)[1:min(
            n_fd_components, length(θ0))]
        for i in fd_idx
            θp = deepcopy(θ0)
            θp[i] = θ0[i] + fd_delta
            simulation.model.trainable_components.θ = θp
            lp = ODINN.loss_iceflow_transient(θp, simulation, map)
            θm = deepcopy(θ0)
            θm[i] = θ0[i] - fd_delta
            simulation.model.trainable_components.θ = θm
            lm = ODINN.loss_iceflow_transient(θm, simulation, map)
            fd = (lp - lm) / (2 * fd_delta)
            relerr = abs(fd - dθ[i]) / max(abs(fd), eps())
            printDebug && @printf("    i=%d  FD=%+.8e  adjoint=%+.8e  relerr=%.2e\n",
                i, fd, dθ[i], relerr)
            @test relerr < thres_fd
        end
        simulation.model.trainable_components.θ = θ0
        return nothing
    end

    ### Computes derivatives with FiniteDifferences.jl (stepsize algorithm included)

    ratio_FD, angle_FD,
    relerr_FD,
    grads_FD = grad_finite_diff(
        simulation; θ = θ, finite_difference_order = finite_difference_order,
        max_params = max_params, mask_parameter_vector = mask_parameter_vector)
    printVecScientific("ratio  = ", [ratio_FD], thres_ratio)
    printVecScientific("angle  = ", [angle_FD], thres_angle)
    printVecScientific("relerr = ", [relerr_FD], thres_relerr)
    if printDebug
        # The three summary statistics cannot distinguish a wrong adjoint from a finite
        # difference that measured noise, so show the magnitudes behind them.
        dθ_adj, dθ_FD = grads_FD
        a, f = collect(dθ_adj), collect(dθ_FD)
        println("  |adjoint| = ", norm(a), "   |FD| = ", norm(f))
        n = min(4, length(a))
        println("  adjoint[1:$n] = ", a[1:n])
        println("  FD[1:$n]      = ", f[1:n])
    end
    @test abs(ratio_FD) < thres_ratio
    @test abs(angle_FD) < thres_angle
    @test abs(relerr_FD) < thres_relerr
end

"""
    test_initial_condition_filter_type_stability()

Solver-free guard: every `initial_condition_filter` must return a concrete float array,
including under Zygote.

`σ_zang` returned the literal `0.0` below its threshold and `x` unchanged above it. Those
agree for a plain `Float64` input, so a direct broadcast looks fine, but inside Zygote the
elements are `ForwardDiff.Dual` and the two branches disagree: `promote_typejoin` widens the
result to `Matrix{Real}`. It only triggers when values sit on *both* sides of the threshold
at once, so it stayed hidden until an optimizer moved some θ.IC across `-β/2` mid-training.

The abstract array then reaches the solver as `u0` via `evaluate_H₀`/`define_iceflow_prob`
and breaks in two unrelated ways: with a stabilized solver OrdinaryDiffEq takes its cache
types from the state's bottom eltype, `one(Real)` is `1::Int64`, and ROCK2 builds a float
tableau as `Int64[...]` (`InexactError`, hundreds of frames away); without one, Zygote
returns no θ.IC gradient at all and the initial condition silently stops training.

No test covered `:Zang1980` on a gradient path -- every gradient test uses `:softplus` -- so
this is checked here directly, with no solver involved.
"""
function test_initial_condition_filter_type_stability()
    # Straddle the threshold: both branches have to be exercised in one broadcast.
    x = [-5.0 0.0; 3.0 -2.0]

    for f in (ODINN.σ_zang, v -> log(1 + exp(v)))
        @test typeof(f.(x)) == Matrix{Float64}

        seen = Ref{Any}(nothing)
        g, = ODINN.Zygote.gradient(x) do t
            y = f.(t)
            ODINN.Zygote.ignore() do
                seen[] = typeof(y)
            end
            sum(y)
        end
        @test isconcretetype(eltype(seen[]))
        @test eltype(seen[]) <: AbstractFloat
        # A widened eltype also costs the gradient outright, so assert it survives.
        @test g isa AbstractMatrix
        @test all(isfinite, g)
    end
end

"""
    test_observation_weights()

Weights of the observations in the loss. Every observation must count: the first one and a
single one used to get weight zero, so a first thickness survey or a single velocity map was
silently ignored.
"""
function test_observation_weights()
    tspan = (2010.0, 2020.0)
    t_obs = [2010.0, 2012.0, 2016.0]

    @test ODINN.observation_weights(:uniform, t_obs, tspan) == [1.0, 1.0, 1.0]
    @test ODINN.observation_weights(:uniform, [2015.0], tspan) == [1.0]

    w = ODINN.observation_weights(:time_span, t_obs, tspan)
    @test w ≈ [1.0, 3.0, 6.0]
    @test sum(w) ≈ tspan[2] - tspan[1]
    @test ODINN.observation_weights(:time_span, [2015.0], tspan) ≈ [10.0]
    # An observation outside tspan is never simulated, so it gets no weight
    @test ODINN.observation_weights(:time_span, [2005.0, 2015.0], tspan) ≈ [0.0, 10.0]
    @test isempty(ODINN.observation_weights(:time_span, Sleipnir.Float[], tspan))
    @test_throws ArgumentError ODINN.observation_weights(:unknown, t_obs, tspan)

    # The choice is read from the loss, also inside MultiLoss and LossHV
    loss = MultiLoss(losses = (LossH(weighting = :time_span), LossV()), λs = (1.0, 1.0))
    @test ODINN.observation_weighting(loss, :H) == :time_span
    @test ODINN.observation_weighting(loss, :V) == :uniform
    @test ODINN.observation_weighting(LossHV(), :H) == :uniform
    @test isnothing(ODINN.observation_weighting(LossH(), :V))
    # Two terms with different weightings of the same observations are refused
    @test_throws ArgumentError ODINN.observation_weighting(
        MultiLoss(losses = (LossH(), LossH(weighting = :time_span)), λs = (1.0, 1.0)), :H)
end

"""
    test_align_start_in_inversion()

When the initial condition is trained, the `Inversion` moves the start of a glacier to its
first ice thickness survey, if the survey is at most one mass balance step after the start.
Otherwise the window is not changed.
"""
function test_align_start_in_inversion()
    for train_initial_conditions in (true, false)
        simulation,
        θ = test_grad_finite_diff(
            ContinuousAdjoint(VJP_method = DiscreteVJP());
            train_initial_conditions = train_initial_conditions, return_setup = true)
        params = simulation.parameters
        glacier = simulation.glaciers[1]
        t_surveys = Sleipnir.tdata(glacier.thicknessData)
        t₁ = params.simulation.tspan[2]
        # The window starts shortly after the first survey, and the second one is a step later
        start = (t_surveys[1] + 0.02, t₁)
        params.simulation.glacier_tspans[glacier.rgi_id] = start
        Inversion(simulation.model, simulation.glaciers, params)
        @test params.simulation.glacier_tspans[glacier.rgi_id] ==
              (train_initial_conditions ? (t_surveys[2], t₁) : start)
    end

    # The regularizations of C and A need these parameters to be trained
    simulation,
    θ = test_grad_finite_diff(
        ContinuousAdjoint(VJP_method = DiscreteVJP());
        functional_inv = false, scalar = false, return_setup = true)
    @test_throws ArgumentError ODINN.check_loss(SlidingRegularization(), simulation)
    @test isnothing(ODINN.check_loss(RheologyRegularization(), simulation))
    # The bed regularization needs a free bed, and the surface loss and a free bed need the
    # automatic adjoint
    @test_throws ArgumentError ODINN.check_loss(BedRegularization(), simulation)
    @test_throws ArgumentError ODINN.check_loss(LossS(), simulation)
    @test_throws ArgumentError test_grad_finite_diff(
        ContinuousAdjoint(VJP_method = DiscreteVJP());
        functional_inv = false, scalar = false, train_initial_conditions = true,
        train_bed = true, return_setup = true)
end

"""
    test_free_bed()

With `InitialCondition(...; bed = :free)` the unknowns are `H₀` and `S₀`, and the bed of the
ice flow model is `S₀ - H₀`. `S₀` starts at the DEM of the glacier, and its change is zero
where the ice is forced to zero. Without θ the bed is `glacier.B`. The surface loss is zero
when the surface is the DEM. A free bed needs a `LossS`.
"""
function test_free_bed()
    loss = MultiLoss(losses = (LossH(), LossS()))
    simulation,
    θ = test_grad_finite_diff(SciMLSensitivityAdjoint();
        functional_inv = false, scalar = false, train_initial_conditions = true,
        train_bed = true, loss = loss, return_setup = true)
    glacier = simulation.glaciers[1]
    H₀ = ODINN.initial_thickness(θ, simulation, 1)
    @test Huginn.bed(simulation, 1, θ) ≈ glacier.S .- H₀
    @test Huginn.bed(simulation, 1, nothing) == glacier.B
    # The outline of the glacier is a hard constraint: no ice outside it, also in the zone
    # where the glacier may advance, and at least `min_thickness` inside it
    outline = glacier.H₀ .> 0
    IC = simulation.model.trainable_components.IC
    @test IC.min_thickness == 1.0
    @test count(.!glacier.mask .& .!outline) > 0
    @test all(H₀[.!outline] .== 0)
    @test all(H₀[outline] .>= IC.min_thickness)
    @test keys(ODINN.ic_params(θ, 1)) == (:H, :S)
    # The bed is computed inside the functions that Enzyme differentiates, which crashed when
    # a call there was dispatched at run time, so it must be type stable
    @test (@inferred ODINN.ic_params(θ, 1)) isa AbstractVector
    # The laws of the gridded A and C read their parameters the same way
    @test (@inferred ODINN.glacier_params(θ.A, 1)) isa AbstractArray
    @test @inferred(ODINN.bed_is_free(θ, 1))
    @test (@inferred Huginn.bed(simulation, 1, θ)) isa Matrix{Float64}
    # `scale_H = 0` (the default) is the maximum thickness of each glacier. With a number, θ.IC
    # is the same thickness divided by that number
    @test (IC.scale_H, IC.scale_S) == (0.0, 5.0)
    ic50 = InitialCondition(simulation.parameters, simulation.glaciers, :Farinotti19;
        bed = :free, scale_H = 50.0, scale_S = 2.0, min_thickness = 3.0)
    @test (ic50.scale_H, ic50.scale_S, ic50.min_thickness) == (50.0, 2.0, 3.0)
    @test typeof(ic50.scale_H) == Sleipnir.Float
    @test_throws ArgumentError InitialCondition(simulation.parameters,
        simulation.glaciers, :Farinotti19; min_thickness = -1.0)
    @test getproperty(ic50.θ.θ, Symbol("1")).H .* 50.0 ≈
          ODINN.ic_params(θ, 1).H .* ODINN.H₀_scale(glacier)
    @test_throws ArgumentError InitialCondition(simulation.parameters,
        simulation.glaciers, :Farinotti19; scale_H = -1.0)
    # The default filter keeps H₀ positive, and building the initial condition with `:identity` warns
    @test UDEparameters().initial_condition_filter == :Zang1980
    p = simulation.parameters
    params_identity = ODINN.Parameters(physical = p.physical, simulation = p.simulation,
        solver = p.solver, hyper = p.hyper,
        UDE = UDEparameters(initial_condition_filter = :identity))
    @test_logs (:warn, r"identity") InitialCondition(
        params_identity, simulation.glaciers, :Farinotti19)
    θ₁ = copy(θ)
    ODINN.ic_params(θ₁, 1).S .= 0.1
    @test Huginn.bed(simulation, 1, θ₁) ≈ glacier.S .- H₀ .+ 0.5 .* .!glacier.mask
    # With each filter, the bed is `S₀ - H₀`: `glacier.S + S × scale_S - H₀` where ice is allowed
    # and `glacier.S` elsewhere
    for filter in (:identity, :softplus, :Zang1980)
        sim = (; glaciers = simulation.glaciers, model = simulation.model,
            parameters = (; UDE = (; initial_condition_filter = filter)))
        scales = ODINN.ic_scales(sim, 1)
        H₀_filter = ODINN.evaluate_H₀(θ₁, glacier, filter, 1;
            scale = scales.H, H_min = scales.Hmin)
        S₀ = glacier.S .+ ODINN.ic_params(θ₁, 1).S .* scales.S .* .!glacier.mask
        @test ODINN.evaluate_B(θ₁, sim, 1) ≈ S₀ .- H₀_filter
        @test all(H₀_filter[.!outline] .== 0)
    end

    # The derivative of the thickness with the minimum, used by the manual adjoints, against
    # finite differences of the thickness. It is zero outside the outline.
    for filter in (:softplus, :Zang1980)
        sim = (; glaciers = simulation.glaciers, model = simulation.model,
            parameters = (; UDE = (; initial_condition_filter = filter)))
        scales = ODINN.ic_scales(sim, 1)
        kw = (; scale = scales.H, H_min = scales.Hmin)
        ε = 1e-6
        θp, θm = copy(θ₁), copy(θ₁)
        ODINN.ic_params(θp, 1).H .+= ε
        ODINN.ic_params(θm, 1).H .-= ε
        fd = (ODINN.evaluate_H₀(θp, glacier, filter, 1; kw...) .-
              ODINN.evaluate_H₀(θm, glacier, filter, 1; kw...)) ./ (2ε)
        ∂H = ODINN.evaluate_∂H₀(θ₁, glacier, filter, 1; kw...)
        @test ∂H≈fd rtol=1e-4 atol=1e-3
        @test all(∂H[.!outline] .== 0)
    end

    # The date of the outline is compared with the start of the run
    @test ODINN.outline_date_status(nothing, 2000.1) == :unknown
    # RGI 6 (2003.6) with NASADEM (2000.1) is the typical case: 3.5 years, no warning
    @test ODINN.outline_date_status(2003.6, 2000.1) == :match
    @test ODINN.outline_date_status(2004.6, 2000.1) == :mismatch
    @test ODINN.outline_date_status(2004.6, 2000.1, 5) == :match
    # Unknown date: information. Far date: warning. Close date: nothing.
    @test_logs (:info, r"not in the data") ODINN.check_outline_date(simulation, [nothing])
    t₀ = glacier_tspan(simulation, 1)[1]
    @test_logs (:warn, r"more than") ODINN.check_outline_date(simulation, [t₀ + 5])
    @test_logs ODINN.check_outline_date(simulation, [t₀ + 1])
    @test length(ODINN.outline_dates(simulation)) == length(simulation.glaciers)

    # The DEMs are the surface at the start and 2 m higher later, so a state that matches
    # them adds nothing to the loss, also with a cell without value
    t = glacier.surfaceData.t
    @test ODINN.time_aggregated_loss(LossS(), [H₀, H₀ .+ 2.0], nothing, nothing, nothing,
        nothing, t, 1, θ, simulation, 1.0, (;)) ≈ 0.0 atol = 1e-12
    @test ODINN.time_aggregated_loss(LossS(), [H₀, H₀], nothing, nothing, nothing,
        nothing, t, 1, θ, simulation, 1.0, (;)) > 0.0

    # The convexity term rewards thicker ice
    reg, reg_convex = BedRegularization(), BedRegularization(convexity = 1.0)
    t₀ = glacier_tspan(simulation, 1)[1]
    R(r) = ODINN.loss(r, H₀, nothing, nothing, nothing, nothing, t₀, 1, θ, simulation, 1.0,
        (; H = 0.0, V = 0.0))
    @test R(reg) > 0
    @test R(reg_convex) < R(reg)

    # The free bed needs the surface loss
    @test_throws ArgumentError test_grad_finite_diff(SciMLSensitivityAdjoint();
        functional_inv = false, scalar = false, train_initial_conditions = true,
        train_bed = true, loss = LossH(), return_setup = true)

    # It also needs a DEM inside the time window. A window after the last DEM has none
    @test isnothing(ODINN.check_loss(LossS(), simulation))
    t₁ = glacier_tspan(simulation, 1)[2]
    simulation.parameters.simulation.glacier_tspans[glacier.rgi_id] = (
        last(glacier.surfaceData.t) + 0.5, t₁)
    @test_throws ArgumentError ODINN.check_loss(LossS(), simulation)
    # Without a free bed, the DEMs are only a loss, so a missing one is a warning
    @test_logs (:warn, r"no DEM inside its time window") ODINN.check_loss(
        LossS(), (; glaciers = simulation.glaciers, parameters = simulation.parameters,
            model = (; trainable_components = (; θ = nothing))))
end

"""
    test_ic_scale_in_batches()

Each batch has one glacier, with index 1, and shares the `InitialCondition` of the model. The
scale of `H₀` must follow the glacier of the batch and not the index: the initial thickness of
a batch is the one of the same glacier in the full simulation.
"""
function test_ic_scale_in_batches()
    simulation,
    θ = test_grad_finite_diff(ContinuousAdjoint(VJP_method = DiscreteVJP());
        multiglacier = true, train_initial_conditions = true, return_setup = true)
    scales = [ODINN.ic_scales(simulation, i).H for i in eachindex(simulation.glaciers)]
    # The test needs glaciers with different thickness
    @test scales[1] != scales[2]
    for i in eachindex(simulation.glaciers)
        batch = ODINN.build_simulation_batch(simulation, i)
        θ_batch = batch.model.trainable_components.θ
        @test ODINN.ic_scales(batch, 1).H == scales[i]
        @test ODINN.initial_thickness(θ_batch, batch, 1) ≈
              ODINN.initial_thickness(θ, simulation, i)
    end
end

"""
    test_loss_terms()

`loss_terms` returns the value of each term of the `MultiLoss`: the weighted terms add up to
the loss of the inversion and the shares add up to 1. The uncertainty of a data term is the one
of the loss, else the one of the data, else the default of the loss.
"""
function test_loss_terms()
    loss = MultiLoss(losses = (LossH(), LossS(), BedRegularization()), λs = (1.0, 0.5, 2.0))
    simulation,
    θ = test_grad_finite_diff(SciMLSensitivityAdjoint();
        functional_inv = false, scalar = false, train_initial_conditions = true,
        train_bed = true, loss = loss, return_setup = true)
    terms = ODINN.loss_terms(θ, simulation)
    @test [typeof(t.term) for t in terms] == [typeof(l) for l in loss.losses]
    @test [t.λ for t in terms] == [1.0, 0.5, 2.0]
    @test sum(t.λ * t.value for t in terms) ≈
          ODINN.loss_iceflow_transient(θ, simulation, map)
    @test sum(t.share for t in terms) ≈ 1.0
    @test all(t -> t.value >= 0, terms)
    # The default weights are 1
    @test MultiLoss(losses = (LossH(), LossS())).λs == (1.0, 1.0)

    # Uncertainty of the thickness: a number of the loss, the data, or the default
    glacier = simulation.glaciers[1]
    t₁ = first(glacier.thicknessData.t)
    @test ODINN.thickness_σ(LossH(), glacier, t₁) == 18.0
    @test ODINN.thickness_σ(LossH(σ = 30.0), glacier, t₁) == 30.0
    σ_map = fill(12.0, size(glacier.H₀))
    σ_map[1, 1] = NaN
    with_σ = Sleipnir.Glacier2D(glacier;
        thicknessData = Sleipnir.ThicknessData(
            t = glacier.thicknessData.t, H = glacier.thicknessData.H,
            σ = [copy(σ_map) for _ in glacier.thicknessData.t]))
    σ = ODINN.thickness_σ(LossH(), with_σ, t₁)
    @test σ[2, 2] == 12.0
    @test σ[1, 1] == 1.0 # not valid, and the cell is not used
    @test ODINN.thickness_σ(LossH(σ = 30.0), with_σ, t₁) == 30.0
    # Uncertainty of the velocity: a number, the error of the map, or the default
    @test ODINN.velocity_σ(LossV(), glacier, 1) == 10.0
    @test ODINN.velocity_σ(LossV(σ = 3.0), glacier, 1) == 3.0
    @test ODINN.velocity_σ(LossAvgV(), glacier, 1) == 10.0
    @test ODINN.default_σ(LossDhdt()) == 0.2
    # With the error of the map (the size of the error vector): 1/√2 of it for the speed and
    # for each component
    n = size(glacier.H₀)
    with_error = Sleipnir.Glacier2D(glacier;
        velocityData = Sleipnir.SurfaceVelocityData(; vx = [zeros(n)], vy = [zeros(n)],
            vabs = [zeros(n)], vabs_error = [6.0], isGridGlacierAligned = true))
    @test ODINN.velocity_σ(LossV(component = :abs), with_error, 1) ≈ 6.0 / sqrt(2)
    @test ODINN.velocity_σ(LossV(component = :xy), with_error, 1) ≈ 6.0 / sqrt(2)
    @test ODINN.velocity_σ(LossAvgV(), with_error, 1) ≈ 6.0 / sqrt(2)
    @test ODINN.velocity_σ(LossV(σ = 3.0), with_error, 1) == 3.0
end

"""
    test_first_observation_counts()

Changing the first ice thickness observation must change the loss. With the old weights,
the time since the previous observation, the first one had weight zero and was ignored.
"""
function test_first_observation_counts()
    simulation,
    θ = test_grad_finite_diff(
        ContinuousAdjoint(VJP_method = DiscreteVJP());
        functional_inv = false, scalar = true, loss = LossH(), return_setup = true)
    H_obs = simulation.glaciers[1].thicknessData.H
    L₀ = ODINN.loss_iceflow_transient(θ, simulation, map)
    # Only the observed cells change, so that the mask of the loss stays the same
    observed = H_obs[1] .> 0
    H_obs[1][observed] .+= 10.0
    L₁ = ODINN.loss_iceflow_transient(θ, simulation, map)
    H_obs[1][observed] .-= 10.0
    # The prediction is close to the observations (ground truth), so +10 m makes the loss larger
    @test L₁ > L₀
end

"""
    test_loss_or_inf()

When the forward simulation fails, the loss is infinite and the line search of LBFGS takes a
shorter step. Other errors must not be hidden. The loss below fails far from the minimum, like
the forward simulation does when the first step of LBFGS is too large.
"""
function test_loss_or_inf()
    out_of_range = ODINN.Muninn.ElevationOutOfRange("diverged")
    @test isinf(ODINN.loss_or_inf(θ -> throw(out_of_range), 1.0))
    @test isinf(ODINN.loss_or_inf(
        θ -> throw(ODINN.ForwardSolveFailure(ODINN.ReturnCode.Unstable)), 1.0))
    # `pmap` wraps the error in a `CapturedException`, also when it runs in the main process
    @test isinf(ODINN.loss_or_inf(θ -> ODINN.pmap(x -> throw(out_of_range), [θ]), 1.0))
    @test_throws CapturedException ODINN.loss_or_inf(
        θ -> ODINN.pmap(x -> x + nothing, [θ]), 1.0)
    @test_throws MethodError ODINN.loss_or_inf(θ -> θ + nothing, 1.0)
    @test_throws DomainError ODINN.loss_or_inf(θ -> sqrt(-θ), 1.0)

    f(x) = any(abs.(x) .> 5) ? throw(out_of_range) : 1e3 * sum(abs2, x .- 1)
    g!(G, x) = (G .= 2e3 .* (x .- 1))
    optimizer = Optim.LBFGS(linesearch = ODINN.LineSearches.BackTracking())
    # The first step, x - g, is at 2000, so without the `Inf` the optimization fails
    @test_throws ODINN.Muninn.ElevationOutOfRange Optim.optimize(f, g!, zeros(2), optimizer)
    res = Optim.optimize(x -> ODINN.loss_or_inf(f, x), g!, zeros(2), optimizer)
    @test Optim.minimum(res) < 1e-8
    @test Optim.minimizer(res) ≈ ones(2) atol = 1e-4
end

"""
    test_loss_sigma()

The simple losses are the mean over the mask of `(misfit / σ)²`: a misfit equal to σ gives 1,
whatever the number of pixels and `normalization`. σ is a number or a matrix. The manual
backward is checked against Zygote, which is what the automatic adjoint differentiates.
"""
function test_loss_sigma()
    nx, ny = 8, 7
    a = [2.0 + 0.3i + 0.1j for i in 1:nx, j in 1:ny]
    b = [1.5 + 0.2i - 0.05j for i in 1:nx, j in 1:ny]
    mask = falses(nx, ny)
    mask[2:5, 3:6] .= true # 16 observed pixels
    normalization = Float64(nx * ny)
    # A misfit of σ everywhere gives 1, with any mask and normalization
    @test loss(L2Sum(), b .+ 3.0, b, mask, normalization; σ = 3.0) ≈ 1.0
    @test loss(L2Sum(), b .+ 3.0, b, trues(nx, ny), 1.0; σ = 3.0) ≈ 1.0
    @test loss(LogSum(σ = 0.1), (b .+ 0.1) .* exp(0.1) .- 0.1, b, mask, 1.0) ≈ 1.0
    # An empty mask gives 0
    @test loss(L2Sum(), a, b, falses(nx, ny), normalization) == 0.0
    σ_map = [1.0 + 0.1i for i in 1:nx, j in 1:ny]
    for (simple, σ) in ((L2Sum(), 2.0), (L2Sum(), σ_map), (LogSum(σ = 0.2), 1.0))
        @test loss(simple, a, b, mask, 1.0; σ = σ) == loss(simple, a, b, mask, 1e6; σ = σ)
        ∂a_zygote, = Zygote.gradient(x -> loss(simple, x, b, mask, normalization; σ = σ), a)
        @test backward_loss(simple, a, b, mask, normalization; σ = σ) ≈ ∂a_zygote
    end
end

"""
    test_loss_time_window_guard()

`LossAvgV` must fail at `Inversion` construction, with a clear message, when the velocity
observation period is not inside `tspan`. Before, it failed deep inside the first gradient
call with `invalid index: nothing`. The same for the period of `LossDhdt` and an explicit `t₀`
of `InitialThicknessRegularization`. Only `tspan`, `rgi_id` and the observation dates are
read, so a small stand-in for the simulation is enough.
"""
function test_loss_time_window_guard()
    vd = (; date1 = [ODINN.Sleipnir.Dates.DateTime(2017, 1, 1)],
        date2 = [ODINN.Sleipnir.Dates.DateTime(2018, 1, 1)])
    sim(tspan) = (;
        parameters = (;
            simulation = (; tspan = tspan,
            glacier_tspans = Dict{String, Tuple{Sleipnir.Float, Sleipnir.Float}}())),
        glaciers = [(;
            rgi_id = "RGI60-11.01450", velocityData = vd, thicknessData = nothing)])
    loss = MultiLoss(losses = (LossH(), LossAvgV()), λs = (1.0, 1.0))

    @test isnothing(ODINN.check_loss(loss, sim((2016.0, 2019.0))))
    @test_throws ArgumentError ODINN.check_loss(loss, sim((2010.0, 2017.5)))
    @test_throws ArgumentError ODINN.check_loss(loss, sim((2017.5, 2019.0)))
    # Losses that need no time window are not affected
    @test isnothing(ODINN.check_loss(LossH(), sim((2010.0, 2012.0))))
    # A wrong option is refused when the inversion is built, not in the first loss call
    @test_throws ArgumentError ODINN.check_loss(
        LossH(weighting = :timespan), sim((2010.0, 2012.0)))
    # A sparse survey is eroded to nothing by the default mask of LossH
    H = zeros(10, 10)
    H[5, 5] = 100.0
    sim_H = (; parameters = sim((2010.0, 2012.0)).parameters,
        glaciers = [(; rgi_id = "RGI60-11.01450", thicknessData = (; H = [H]))])
    @test_logs (:warn,) ODINN.check_loss(LossH(), sim_H)
    @test_logs ODINN.check_loss(LossH(loss = L2Sum(distance = 0)), sim_H)

    # The elevation change period and an explicit t₀ of the H₀ regularization too
    sim_dhdt(tspan) = (; parameters = sim(tspan).parameters,
        glaciers = [(; rgi_id = "RGI60-11.01450", dhdtData = (; t = (2010.0, 2015.0)))])
    @test isnothing(ODINN.check_loss(LossDhdt(), sim_dhdt((2010.0, 2015.0))))
    @test_throws ArgumentError ODINN.check_loss(
        LossDhdt(), sim_dhdt((2011.0, 2015.0)))
    @test isnothing(ODINN.check_loss(
        InitialThicknessRegularization(), sim((2016.0, 2019.0))))
    @test isnothing(ODINN.check_loss(
        InitialThicknessRegularization(2016.0), sim((2016.0, 2019.0))))
    @test_throws ArgumentError ODINN.check_loss(
        InitialThicknessRegularization(2010.0), sim((2016.0, 2019.0)))
end

"""
    test_initial_thickness_regularization_backward()

Check the manual `backward_loss` of `InitialThicknessRegularization` against Zygote and a
central finite difference of its own forward `loss`, with no solver in the comparison.

`θ.IC` is stored scaled by `H₀_scale`, so `∂L/∂θ.IC = ∂L/∂H₀ ⋅ ∂H₀/∂θ.IC`. The manual
backward used to return `∂L/∂H₀` directly. That was only right where the filter derivative
is 1 (thick ice) and, with the scaling, it is `H₀_scale` times too small everywhere. The joint
A + IC test in Core8 cannot see it: the regularization is a small part of that gradient and
the finite difference only samples 60 of its components, so it passes with the bug.
"""
function test_initial_thickness_regularization_backward()
    simulation,
    θ = test_grad_finite_diff(
        ContinuousAdjoint(VJP_method = DiscreteVJP());
        functional_inv = false, scalar = false, train_initial_conditions = true,
        loss = MultiLoss(
            losses = (LossH(), InitialThicknessRegularization(2010.0)), λs = (1.0, 1.0)),
        return_setup = true)
    glacier = simulation.glaciers[1]
    # With a scale near 1 a missing chain-rule factor would go unnoticed
    @test ODINN.H₀_scale(glacier) > 10

    reg = InitialThicknessRegularization(2010.0)
    nrm = Sleipnir.Float(prod(size(glacier.H₀)))
    Δtj = (; H = 0.0, V = 0.0)
    fwd(p) = ODINN.loss(reg, glacier.H₀, nothing, nothing, nothing, nothing,
        2010.0, 1, p, simulation, nrm, Δtj)
    manual = ODINN.backward_loss(reg, glacier.H₀, nothing, nothing, nothing, nothing,
        2010.0, 1, θ, simulation, nrm, Δtj)[2]
    zygote, = ODINN.Zygote.gradient(fwd, θ)

    # getproperty returns a view, θ.IC[sym] would return a copy
    icview(p) = vec(ODINN.ic_params(p, 1).H)
    m, z = icview(manual), icview(zygote)
    @test norm(m .- z) / norm(z) < 1e-10

    for k in sortperm(abs.(z); rev = true)[1:5]
        h = 1e-6 * max(abs(icview(θ)[k]), 1.0)
        pp = copy(θ)
        icview(pp)[k] += h
        pm = copy(θ)
        icview(pm)[k] -= h
        fd = (fwd(pp) - fwd(pm)) / (2h)
        @test abs(m[k] - fd) / abs(fd) < 1e-6
    end
end

"""
    test_sliding_regularization_backward()

Check the manual `backward_loss` of `SlidingRegularization` (through the tanh map of `LawC`)
against Zygote and a central finite difference of its own forward `loss`, with no solver.
"""
function test_sliding_regularization_backward()
    simulation,
    θ = test_grad_finite_diff(
        ContinuousAdjoint(VJP_method = DiscreteVJP());
        target = :C, functional_inv = false, scalar = false, maxC = 0.1,
        return_setup = true)
    glacier = simulation.glaciers[1]
    Cview(p) = vec(getproperty(p.C, Symbol("1")))
    # A constant C has no Laplacian, so move θ.C away from its seed
    Cview(θ) .+= 0.5 .* randn(MersenneTwister(1), length(Cview(θ)))

    reg = SlidingRegularization()
    t₀ = simulation.parameters.simulation.tspan[1]
    nrm = Sleipnir.Float(prod(size(glacier.H₀)))
    Δtj = (; H = 0.0, V = 0.0)
    fwd(p) = ODINN.loss(reg, glacier.H₀, nothing, nothing, nothing, nothing,
        t₀, 1, p, simulation, nrm, Δtj)
    manual = ODINN.backward_loss(reg, glacier.H₀, nothing, nothing, nothing, nothing,
        t₀, 1, θ, simulation, nrm, Δtj)[2]
    zygote, = ODINN.Zygote.gradient(fwd, θ)
    m, z = Cview(manual), Cview(zygote)
    @test norm(m .- z) / norm(z) < 1e-10

    for k in sortperm(abs.(z); rev = true)[1:5]
        h = 1e-6 * max(abs(Cview(θ)[k]), 1.0)
        pp = copy(θ)
        Cview(pp)[k] += h
        pm = copy(θ)
        Cview(pm)[k] -= h
        fd = (fwd(pp) - fwd(pm)) / (2h)
        @test abs(m[k] - fd) / abs(fd) < 1e-6
    end
end

"""
    test_grad_V_from_Vxy()

Solver-free finite-difference check for the `:abs` component of the velocity losses
(`LossV` / `LossAvgV`). The `:abs` branch builds the loss from the velocity magnitude
`V = √(Vx² + Vy²)` and must propagate the gradient back to `(Vx, Vy)` with the exact
chain rule `∂ℓ/∂Vx = ∂ℓ/∂V · Vx/V`. This reproduces that branch with the **real** loss
functions (`loss` / `backward_loss` of the simple loss, composed with `ODINN.VJP_λ_∂V∂Vxy`)
and compares to finite differences of the same velocity loss.

It guards against regressing to the previous secant-slope form
`∂ℓ/∂V · (Vx_pred - Vx_ref)/(V_pred - V_ref)`, which equals the chain rule only when the
predicted and reference flow directions coincide — a condition that holds closely enough
for SIA that no integration test reliably exposes the error (the secant blows up as
`V_pred → V_ref`). Here FD is exact and the secant would fail by O(1).
"""
function test_grad_V_from_Vxy()
    nx, ny = 6, 5
    # Deterministic, strictly-positive fields; predicted and reference flow directions
    # deliberately differ so the (buggy) secant slope departs from the true chain rule.
    Vx_pred = [1.0 + 0.30i + 0.10j for i in 1:nx, j in 1:ny]
    Vy_pred = [2.0 + 0.20i - 0.05j for i in 1:nx, j in 1:ny]
    Vx_ref = [1.2 + 0.10i - 0.05j for i in 1:nx, j in 1:ny]
    Vy_ref = [1.5 - 0.15i + 0.10j for i in 1:nx, j in 1:ny]
    V_ref = sqrt.(Vx_ref .^ 2 .+ Vy_ref .^ 2)
    mask = trues(nx, ny)
    normalization = 3.0

    for simpleLoss in (L2Sum(), LogSum())
        # Effective :abs velocity loss as a function of (Vx, Vy), using the real loss fn
        L(vx, vy) = loss(simpleLoss, sqrt.(vx .^ 2 .+ vy .^ 2), V_ref, mask, normalization)

        # Adjoint, composed exactly as the `:abs` branch does
        V_pred = sqrt.(Vx_pred .^ 2 .+ Vy_pred .^ 2)
        ∂l∂V = backward_loss(simpleLoss, V_pred, V_ref, mask, normalization)
        ∂Vx, ∂Vy = ODINN.VJP_λ_∂V∂Vxy(∂l∂V, Vx_pred, Vy_pred)

        ∂Vx_FD, ∂Vy_FD = FiniteDifferences.grad(central_fdm(5, 1), L, Vx_pred, Vy_pred)

        thres = 1e-9
        for (a, b) in ((∂Vx, ∂Vx_FD), (∂Vy, ∂Vy_FD))
            ratio, angle, relerr = stats_err_arrays(a, b)
            if printDebug |
               !((abs(ratio) < thres) & (abs(angle) < thres) & (abs(relerr) < thres))
                printVecScientific("ratio  = ", [ratio], thres)
                printVecScientific("angle  = ", [angle], thres)
                printVecScientific("relerr = ", [relerr], thres)
            end
            @test (abs(ratio) < thres) & (abs(angle) < thres) & (abs(relerr) < thres)
        end
    end
end

function test_grad_L2Sum()
    function _loss!(l, a, b, norm, lossType)
        l[1] = loss(lossType, a, b, norm)
        return nothing
    end

    lossType = L2Sum(distance = 2)
    nx = 9
    ny = 10
    norm = 3.5
    a = randn(nx, ny)
    a[1, :] .= 0;
    a[end, :] .= 0;
    a[:, 1] .= 0;
    a[:, end] .= 0
    b = randn(nx, ny)
    b[1, :] .= 0;
    b[end, :] .= 0;
    b[:, 1] .= 0;
    b[:, end] .= 0
    l = [0.0]
    _loss!(l, a, b, norm, lossType)
    dl_enzyme = [1.0]
    l_enzyme = Enzyme.make_zero(dl_enzyme)
    da_enzyme = Enzyme.make_zero(a)
    Enzyme.autodiff(
        set_runtime_activity(EnzymeCore.Reverse), _loss!, Const,
        Duplicated(l_enzyme, dl_enzyme),
        Duplicated(a, da_enzyme),
        Enzyme.Const(b),
        Enzyme.Const(norm),
        Enzyme.Const(lossType)
    )
    da = backward_loss(lossType, a, b, norm)
    ratio, angle, relerr = stats_err_arrays(da, da_enzyme)
    thres = 1e-14
    if printDebug | !((abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres))
        printVecScientific("ratio  = ", [ratio], thres)
        printVecScientific("angle  = ", [angle], thres)
        printVecScientific("relerr = ", [relerr], thres)
    end
    @test (abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres)
end

function test_grad_TikhonovRegularization()
    function _loss!(l, a, Δx, Δy, mask, norm, lossType)
        l[1] = loss(lossType, a, Δx, Δy, mask, norm)
        return nothing
    end

    nx = 9
    ny = 10
    norm = 3.5
    a = randn(nx, ny)
    a[1:2, :] .= 0;
    a[(end - 1):end, :] .= 0;
    a[:, 1:2] .= 0;
    a[:, (end - 1):end] .= 0
    Δx = 1.2
    Δy = 1.8
    mask = randn(MersenneTwister(1), nx, ny) .>= 0
    for lossType in (TikhonovRegularization(σ = 0.7),
        TikhonovRegularization(operator = :laplacian, σ = 1.3))
        l = [0.0]
        _loss!(l, a, Δx, Δy, mask, norm, lossType)
        dl_enzyme = [1.0]
        l_enzyme = Enzyme.make_zero(dl_enzyme)
        da_enzyme = Enzyme.make_zero(a)
        Enzyme.autodiff(
            set_runtime_activity(EnzymeCore.Reverse), _loss!, Const,
            Duplicated(l_enzyme, dl_enzyme),
            Duplicated(a, da_enzyme),
            Enzyme.Const(Δx),
            Enzyme.Const(Δy),
            Enzyme.Const(mask),
            Enzyme.Const(norm),
            Enzyme.Const(lossType)
        )
        da = backward_loss(lossType, a, Δx, Δy, mask, norm)
        # Zygote is what the automatic adjoint uses for the losses
        da_zygote, = Zygote.gradient(x -> loss(lossType, x, Δx, Δy, mask, norm), a)
        for other in (da_enzyme, da_zygote)
            ratio, angle, relerr = stats_err_arrays(da, other)
            thres = 1e-14
            if printDebug | !((abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres))
                printVecScientific("ratio  = ", [ratio], thres)
                printVecScientific("angle  = ", [angle], thres)
                printVecScientific("relerr = ", [relerr], thres)
            end
            @test (abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres)
        end
    end

    # A plane with slopes 0.01 and 0.02 m/m costs (0.01² + 0.02²) / σ² at any resolution and
    # with any mask
    reg = TikhonovRegularization(σ = 0.1)
    plane(n, Δ) = [0.01 * i * Δ + 0.02 * j * Δ for i in 1:n, j in 1:n]
    expected = (0.01^2 + 0.02^2) / 0.1^2
    @test loss(reg, plane(10, 100.0), 100.0, 100.0, trues(10, 10), 1.0) ≈ expected
    @test loss(reg, plane(20, 50.0), 50.0, 50.0, trues(20, 20), 1.0) ≈ expected
    @test loss(reg, plane(20, 50.0), 50.0, 50.0, mask_corner(20), 1.0) ≈ expected
    @test_throws ArgumentError TikhonovRegularization(operator = :other)
end
mask_corner(n) = BitMatrix([i + j <= n for i in 1:n, j in 1:n])

function _loss_halfar!(l, R₀, h₀, r₀, A, n, tstops, H_ref, params, lossType, glacier, θ)
    normalization = 1.0
    l_H = 0.0
    Δt = diff(tstops)
    physicalParams = params.physical
    for τ in range(2, length(tstops))
        t₁ = tstops[τ]
        _H₁ = halfar_solution(R₀, t₁, h₀, r₀, A[1], n, physicalParams)
        mean_error, = loss(
            lossType,
            _H₁,
            H_ref[τ],
            t = t₁,
            glacier = glacier,
            θ = θ,
            params = params,
            prod(size(H_ref[τ]))/normalization
        )
        l_H += Δt[τ - 1] * mean_error
    end
    l[1] = l_H
    return nothing
end

function test_grad_Halfar(
        adjointFlavor::ADJ;
        thres = [0.0, 0.0, 0.0]
) where {ADJ <: AbstractAdjointMethod}
    lossType = LossH(L2Sum(distance = 15))
    A = 8e-19
    t₀ = 5.0
    t₁ = 30.0
    h₀ = 500
    r₀ = 1000
    n = 3.0
    Δx = 50.0
    Δy = 50.0
    nx = 100
    ny = 100
    δt = 1/12
    T = 2.0
    tstops = collect(t₀:δt:t₁)

    # Get parameters for a simulation
    parameters = Parameters(
        simulation = SimulationParameters(
            tspan = (t₀, t₁),
            multiprocessing = false,
            use_MB = false,
            use_iceflow = true,
            test_mode = true,
            working_dir = Huginn.root_dir
        ),
        physical = PhysicalParameters(
            minA = 8e-21,
            maxA = 8e-18),
        UDE = UDEparameters(
            optim_autoAD = ODINN.NoAD(),
            grad = adjointFlavor,
            optimization_method = "AD+AD",
            empirical_loss_function = lossType,
            target = :A
        ),
        solver = SolverParameters(
            reltol = 1e-12,
            step = δt,
            progress = true
        )
    )

    # Bed (it has to be flat for the Halfar solution)
    B = zeros((nx, ny))

    # Use a constant A for testing
    model = Model(
        iceflow = SIA2Dmodel(parameters),#; A=A_law),
        mass_balance = nothing,
        trainable_components = NeuralNetwork(parameters)
    )

    θ = model.trainable_components.θ
    modelNN = model.trainable_components.architecture
    st = model.trainable_components.st
    smodel = StatefulLuxLayer{true}(modelNN, θ.θ, st)
    min_NN = parameters.physical.minA
    max_NN = parameters.physical.maxA
    A_θ = ODINN.predict_A̅(smodel, [T], (min_NN, max_NN))[1]
    println("A_θ = ", A_θ)

    # Initial condition of the glacier
    R₀ = [sqrt((Δx * (i - nx/2))^2 + (Δy * (j - ny/2))^2) for i in 1:nx, j in 1:ny]
    H₀ = halfar_solution(R₀, t₀, h₀, r₀, A_θ, n, parameters.physical)
    S = B + H₀

    # Define glacier object
    climate = Sleipnir.DummyClimate2D(longterm_temps_scalar = [T], longterm_temps_gridded = [T T;
                                                                                             T T])
    glacier = Glacier2D(
        rgi_id = "toy", climate = climate, H₀ = H₀, S = S, B = B, A = A, n = n,
        Δx = Δx, Δy = Δy, nx = nx, ny = ny, C = 0.0)
    glaciers = Vector{Sleipnir.AbstractGlacier}([glacier])

    fakeA(T) = A
    # TODO: add law
    glaciers = generate_ground_truth(glaciers, parameters, model, tstops)

    model.iceflow = SIA2Dmodel(parameters)

    # We create an ODINN prediction
    simulation = Inversion(model, glaciers, parameters)

    # Compute gradient of with Halfar solution wrt A
    A_θ = [A_θ]
    ∂A_enzyme = Enzyme.make_zero(A_θ)
    dl_enzyme = [1.0]
    l_enzyme = Enzyme.make_zero(dl_enzyme)
    H_ref = simulation.glaciers[1].thicknessData.H
    Enzyme.autodiff(
        EnzymeCore.Reverse, _loss_halfar!, Const,
        Duplicated(l_enzyme, dl_enzyme),
        Enzyme.Const(R₀),
        Enzyme.Const(h₀),
        Enzyme.Const(r₀),
        Duplicated(A_θ, ∂A_enzyme),
        Enzyme.Const(n),
        Enzyme.Const(tstops),
        Enzyme.Const(H_ref),
        Enzyme.Const(parameters),
        Enzyme.Const(lossType),
        Enzyme.Const(glacier),
        Enzyme.Const(θ)
    )

    println("l_enzyme=", l_enzyme)
    println("∂A_enzyme=", ∂A_enzyme)

    # Retrieve apply parametrization from inversion
    # TODO: replace function below
    ∇θ, = Zygote.gradient(
        _θ -> apply_parametrization(
            model.trainable_components.target;
            H = nothing, ∇S = nothing, θ = _θ,
            iceflow_model = only(model.iceflow), trainable_components = model.trainable_components,
            glacier = only(glaciers), params = parameters),
        θ)
    dθ_halfar = ∂A_enzyme[1] * ∇θ

    # Compute gradient with manual implementation of the backward + discrete adjoint of SIA2D
    dθ = zero(θ)
    SIA2D_grad!(dθ, θ, simulation)

    ratio, angle, relerr = stats_err_arrays(dθ, dθ_halfar)

    thres_ratio = thres[1]
    thres_angle = thres[2]
    thres_relerr = thres[3]
    if printDebug |
       !((abs(ratio)<thres_ratio) & (abs(angle)<thres_angle) & (abs(relerr)<thres_relerr))
        printVecScientific("ratio  = ", [ratio], thres_ratio)
        printVecScientific("angle  = ", [angle], thres_angle)
        printVecScientific("relerr = ", [relerr], thres_relerr)
    end
    @test abs(ratio) < thres_ratio
    @test abs(angle) < thres_angle
    @test abs(relerr) < thres_relerr
end

"""
    test_grad_sciml_vs_manual(; thres)

Check that the SciMLSensitivity auto-adjoint and the manual `DiscreteAdjoint` return
consistent gradients for `LawA`. Both methods are run with identical parameters;
only the `grad` field of `UDEparameters` differs.
"""
function test_grad_sciml_vs_manual(; thres = [1e-3, 1e-13, 1e-3])
    println("> Comparing SciMLSensitivity auto-adjoint vs manual ContinuousAdjoint for LawA")

    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()
    working_dir = joinpath(ODINN.root_dir, "test/data")
    δt = 1/12
    tspan = (2010.0, 2012.0)
    tstops = collect(tspan[1]:δt:tspan[2])

    params_sciml = Parameters(
        simulation = SimulationParameters(
            working_dir = working_dir,
            use_MB = false,
            use_velocities = true,
            tspan = tspan,
            step_MB = δt,
            multiprocessing = false,
            workers = 1,
            test_mode = true,
            rgi_paths = rgi_paths,
            gridScalingFactor = 4,
            f_surface_velocity_factor = 0.8
        ),
        hyper = Hyperparameters(batch_size = 1, epochs = 100, optimizer = ODINN.Adam(0.005)),
        physical = PhysicalParameters(minA = 2e-18, maxA = 8e-18),
        UDE = UDEparameters(
            grad = ODINN.SciMLSensitivityAdjoint(),
            sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.EnzymeVJP()),
            optim_autoAD = Optimization.AutoZygote(),
            empirical_loss_function = LossH(),
            target = :A
        ),
        solver = Huginn.SolverParameters(
            step = δt, solver = ROCK4(), supply_eigen_est = true)
    )

    # Identical params except for the adjoint method
    params_manual = Parameters(
        simulation = params_sciml.simulation,
        hyper = params_sciml.hyper,
        physical = params_sciml.physical,
        UDE = UDEparameters(
            sensealg = params_sciml.UDE.sensealg,
            optim_autoAD = params_sciml.UDE.optim_autoAD,
            grad = ContinuousAdjoint(),
            optimization_method = params_sciml.UDE.optimization_method,
            empirical_loss_function = params_sciml.UDE.empirical_loss_function,
            target = params_sciml.UDE.target,
            initial_condition_filter = params_sciml.UDE.initial_condition_filter
        ),
        solver = params_sciml.solver
    )

    # Ground truth using a known constant A
    model_gt = Model(
        iceflow = SIA2Dmodel(params_sciml; A = ConstantA(2.21e-18)),
        mass_balance = TImodel1(params_sciml; DDF = 6.0/1000.0, prcp_fac = 1.2)
    )
    glaciers = initialize_glaciers(rgi_ids, params_sciml)
    glaciers = generate_ground_truth(glaciers, params_sciml, model_gt, tstops)

    # Single NN shared across both simulations so both start at the same θ
    nn_model = NeuralNetwork(params_sciml)
    mass_balance = TImodel1(params_sciml; DDF = 6.0/1000.0, prcp_fac = 1.2)

    sim_sciml = Inversion(
        Model(
            iceflow = SIA2Dmodel(params_sciml; A = LawA(nn_model, params_sciml)),
            mass_balance = mass_balance,
            regressors = (; A = nn_model)
        ),
        glaciers, params_sciml
    )
    θ = sim_sciml.model.trainable_components.θ

    sim_manual = Inversion(
        Model(
            iceflow = SIA2Dmodel(params_manual; A = LawA(nn_model, params_manual)),
            mass_balance = mass_balance,
            regressors = (; A = nn_model)
        ),
        glaciers, params_manual
    )
    sim_manual.model.trainable_components.θ .= θ

    dθ_sciml = ODINN.grad_loss_iceflow!(θ, sim_sciml, map)
    @assert !any(isnan, dθ_sciml) "SciML gradient contains NaNs"

    dθ_manual = zero(θ)
    SIA2D_grad!(dθ_manual, θ, sim_manual)

    ratio, angle, relerr = stats_err_arrays(dθ_sciml, dθ_manual)
    printVecScientific("ratio  = ", [ratio], thres[1])
    printVecScientific("angle  = ", [angle], thres[2])
    printVecScientific("relerr = ", [relerr], thres[3])
    @test abs(ratio) < thres[1]
    @test abs(angle) < thres[2]
    @test abs(relerr) < thres[3]
end
