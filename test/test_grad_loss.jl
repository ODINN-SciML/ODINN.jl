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
        use_MB = false,
        temp_bias = 0.0,
        calibrate_MB = false,
        abstol = 1e-6,
        solver = nothing,
        A_range = nothing,
        # `PhysicalParameters` defaults `maxC` to a value that only makes sense for a Weertman
        # law; against the Budd law the glaciers are built with it is some fifteen orders too
        # small, sliding contributes nothing, and a C gradient sits at round off. Any target
        # `:C` test has to set this to something the configuration actually activates.
        maxC = Sleipnir.PhysicalParameters().maxC,
        adaptive = true,
        dt = 1.0/120.0,
        fd_delta = nothing,
        thres_fd = 5e-2,
        n_fd_components = 4,
        return_grad = false,
        # Return the built `(simulation, θ)` instead of running the gradient, so that a
        # component of the adjoint can be checked in isolation against the very configuration
        # under test rather than a hand-copied replica of this setup, which drifts.
        return_setup = false,
        # Override the sensealg the automatic adjoint uses, to compare VJP backends for the ODE
        # adjoint. Velocity losses are 14.8x wrong through the automatic path while thickness
        # losses are exact in the same configuration, and the one structural difference is that
        # velocity seeds come from an rrule whose pullback runs Enzyme inside the Zygote
        # pullback SciMLSensitivity is orchestrating.
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
            # `UDE.target` is consumed only by the manual adjoints, to dispatch their
            # target-specific VJPs; the automatic adjoint ignores it entirely. A C inversion is
            # defined by its regressor and law, and its trainable components still carry
            # `SIA2D_A_target`, so this has to stay `:A` to satisfy the `Inversion` constructor
            # assertion. Setting it to `:C` is wrong.
            target = target == :C ? :A : target,
            initial_condition_filter = :softplus
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

    ic = train_initial_conditions ? InitialCondition(params, glaciers, :Farinotti19) :
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
        # Classical only: `LawC` defines no `p_VJP!`, so the manual adjoints would silently
        # return a zero gradient for θ.C and there is nothing to test there.
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
        # Central difference at a fixed step, against the adjoint computed under the same
        # solver. FiniteDifferences' adaptive step is unusable here: with an adaptive
        # integrator the loss is discontinuous in θ, so the step it settles on measures
        # step-acceptance jitter rather than a derivative. This path therefore requires
        # `adaptive = false`, where the trajectory depends smoothly on θ.
        @assert !adaptive "A fixed step finite difference is only meaningful with adaptive = false."
        θ0 = deepcopy(θ)

        # A gradient with no signal passes every relative comparison below, because the finite
        # difference is zero there too — that is exactly how a dead gradient hides. Require it
        # to be resolvable against the loss scale rather than merely nonzero: a parameter the
        # configuration barely activates lands at round off, which is indistinguishable from a
        # broken adjoint and equally useless as a test. A gridded C with the default `maxC`
        # against a Budd law is one such case, off by some fifteen orders.
        l0 = f(θ0, simulation)
        @test maximum(abs, collect(dθ)) > sqrt(eps(Float64)) * max(one(Float64), abs(l0))

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
    @test isempty(ODINN.observation_weights(:time_span, Float64[], tspan))
    @test_throws ArgumentError ODINN.observation_weights(:unknown, t_obs, tspan)

    # The choice is read from the loss, also inside MultiLoss and LossHV
    loss = MultiLoss(losses = (LossH(weighting = :time_span), LossV()), λs = (1.0, 1.0))
    @test ODINN.observation_weighting(loss, :H) == :time_span
    @test ODINN.observation_weighting(loss, :V) == :uniform
    @test ODINN.observation_weighting(LossHV(), :H) == :uniform
    @test isnothing(ODINN.observation_weighting(LossH(), :V))
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
    H_obs[1] .+= 10.0
    L₁ = ODINN.loss_iceflow_transient(θ, simulation, map)
    H_obs[1] .-= 10.0
    @test L₁ > L₀
end

"""
    test_loss_or_inf()

When the forward simulation fails, the loss is infinite and the line search of LBFGS takes a
shorter step. Other errors must not be hidden. The loss below fails far from the minimum, like
the forward simulation does when the first step of LBFGS is too large.
"""
function test_loss_or_inf()
    solver_error = AssertionError("There was an error in the iceflow solver. Returned code is \"Unstable\"")
    @test isinf(ODINN.loss_or_inf(θ -> throw(DomainError(θ, "out of range")), 1.0))
    @test isinf(ODINN.loss_or_inf(θ -> throw(solver_error), 1.0))
    # `pmap` wraps the error in a `CapturedException`, also when it runs in the main process
    @test isinf(ODINN.loss_or_inf(θ -> ODINN.pmap(x -> throw(DomainError(x, "diverged")), [θ]), 1.0))
    @test_throws CapturedException ODINN.loss_or_inf(
        θ -> ODINN.pmap(x -> x + nothing, [θ]), 1.0)
    @test_throws MethodError ODINN.loss_or_inf(θ -> θ + nothing, 1.0)
    @test_throws AssertionError ODINN.loss_or_inf(θ -> @assert(false, "other"), 1.0)

    f(x) = any(abs.(x) .> 5) ? throw(DomainError(x, "diverged")) : 1e3 * sum(abs2, x .- 1)
    g!(G, x) = (G .= 2e3 .* (x .- 1))
    optimizer = Optim.LBFGS(linesearch = ODINN.LineSearches.BackTracking())
    # The first step, x - g, is at 2000, so without the `Inf` the optimization fails
    @test_throws DomainError Optim.optimize(f, g!, zeros(2), optimizer)
    res = Optim.optimize(x -> ODINN.loss_or_inf(f, x), g!, zeros(2), optimizer)
    @test Optim.minimum(res) < 1e-8
    @test Optim.minimizer(res) ≈ ones(2) atol = 1e-4
end

"""
    test_loss_time_window_guard()

`LossAvgV` must fail at `Inversion` construction, with a clear message, when the velocity
observation period is not inside `tspan`. Before, it failed deep inside the first gradient
call with `invalid index: nothing`. Only `tspan`, `rgi_id` and the velocity dates are read,
so a small stand-in for the simulation is enough.
"""
function test_loss_time_window_guard()
    vd = (; date1 = [ODINN.Sleipnir.Dates.DateTime(2017, 1, 1)],
        date2 = [ODINN.Sleipnir.Dates.DateTime(2018, 1, 1)])
    sim(tspan) = (; parameters = (; simulation = (; tspan = tspan)),
        glaciers = [(; rgi_id = "RGI60-11.01450", velocityData = vd)])
    loss = MultiLoss(losses = (LossH(), LossAvgV()), λs = (1.0, 1.0))

    @test isnothing(ODINN.check_loss_time_window(loss, sim((2016.0, 2019.0))))
    @test_throws ArgumentError ODINN.check_loss_time_window(loss, sim((2010.0, 2017.5)))
    @test_throws ArgumentError ODINN.check_loss_time_window(loss, sim((2017.5, 2019.0)))
    # Losses that need no time window are not affected
    @test isnothing(ODINN.check_loss_time_window(LossH(), sim((2010.0, 2012.0))))
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
    nrm = Float64(prod(size(glacier.H₀)))
    Δtj = (; H = 0.0, V = 0.0)
    fwd(p) = ODINN.loss(reg, glacier.H₀, nothing, nothing, nothing, nothing,
        2010.0, 1, p, simulation, nrm, Δtj)
    manual = ODINN.backward_loss(reg, glacier.H₀, nothing, nothing, nothing, nothing,
        2010.0, 1, θ, simulation, nrm, Δtj)[2]
    zygote, = ODINN.Zygote.gradient(fwd, θ)

    # getproperty returns a view, θ.IC[sym] would return a copy
    icview(p) = vec(getproperty(p.IC, Symbol("1")))
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

    lossType = TikhonovRegularization()
    nx = 9
    ny = 10
    norm = 3.5
    a = randn(nx, ny)
    a[1:2, :] .= 0;
    a[(end - 1):end, :] .= 0;
    a[:, 1:2] .= 0;
    a[:, (end - 1):end] .= 0
    b = randn(nx, ny)
    b[1:2, :] .= 0;
    b[(end - 1):end, :] .= 0;
    b[:, 1:2] .= 0;
    b[:, (end - 1):end] .= 0
    Δx = 1.2
    Δy = 1.8
    mask = randn(nx, ny) .>= 0
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
    ratio, angle, relerr = stats_err_arrays(da, da_enzyme)
    thres = 1e-14
    if printDebug | !((abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres))
        printVecScientific("ratio  = ", [ratio], thres)
        printVecScientific("angle  = ", [angle], thres)
        printVecScientific("relerr = ", [relerr], thres)
    end
    @test (abs(ratio)<thres) & (abs(angle)<thres) & (abs(relerr)<thres)
end

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
