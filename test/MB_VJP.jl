
function test_MB_VJP(
    VJPMode::AbstractVJPMethod;
    thres = [2e-4, 1e-4, 1e-2],
)

    Random.seed!(1234)

    thres_ratio = thres[1]
    thres_angle = thres[2]
    thres_relerr = thres[3]

    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()

    working_dir = joinpath(ODINN.root_dir, "test/data")

    δt = 1/12
    tspan = (2010.0, 2012.0)

    params = Parameters(
        simulation = SimulationParameters(
            working_dir=working_dir,
            use_MB=true,
            use_velocities=false,
            tspan=tspan,
            step_MB=δt,
            multiprocessing=false,
            test_mode=true,
            rgi_paths=rgi_paths,
            gridScalingFactor=4),
        UDE = UDEparameters(
            sensealg=SciMLSensitivity.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=DiscreteAdjoint(VJP_method=VJPMode)),
        solver = Huginn.SolverParameters(step=δt),
        physical = PhysicalParameters(minA=0.0, maxA=0.0), # Disable creeping by multiplying by zero the NN output that predicts A
    )

    nn_model = NeuralNetwork(params)

    model = Model(
        iceflow = SIA2Dmodel(params; A=LawA(nn_model, params)),
        mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
        regressors = (; A=nn_model),
    )

    glaciers = initialize_glaciers(rgi_ids, params)
    glacier_idx = 1
    glacier = glaciers[glacier_idx]

    H = glaciers[glacier_idx].H₀

    simulation = Inversion(model, glaciers, params)
    θ = simulation.model.trainable_components.θ
    simulation.cache = init_cache(model, simulation, glacier_idx, θ)

    t = mean(tspan)
    λ = randn(size(H, 1), size(H, 2))
    ∂H = ODINN.VJP_λ_∂MB∂H(VJPMode, λ, H, simulation, glacier, t) + λ

    # Check gradient wrt H
    function f_H(H, args)
        simulation, t, step, λ, glacier = args
        model = simulation.model
        cache = simulation.cache
        glacier.S .= glacier.B .+ H
        MB_timestep!(cache, model, glacier, step, t)
        apply_MB_mask!(H, cache.iceflow)
        return sum(H.*λ)
    end
    ratio = []
    angle = []
    relerr = []
    eps = []
    for k in range(-5,1,step=2)
        ϵ = 10.0^(-k)
        push!(eps, ϵ)
        ∂H_num = compute_numerical_gradient(H, (simulation, t, δt, λ, glacier), f_H, ϵ; varStr="of H")
        ratio_k, angle_k, relerr_k = stats_err_arrays(∂H, ∂H_num)
        push!(ratio, ratio_k)
        push!(angle, angle_k)
        push!(relerr, relerr_k)
    end
    min_ratio = minimum(abs.(ratio))
    min_angle = minimum(abs.(angle))
    min_relerr = minimum(abs.(relerr))
    if printDebug | !( (min_ratio<thres_ratio) & (min_angle<thres_angle) & (min_relerr<thres_relerr) )
        println("Gradient wrt H")
        println("eps    = ",printVecScientific(eps))
        printVecScientific("ratio  = ",ratio,thres_ratio)
        printVecScientific("angle  = ",angle,thres_angle)
        printVecScientific("relerr = ",relerr,thres_relerr)
    end
    @test min_ratio<thres_ratio
    @test min_angle<thres_angle
    @test min_relerr<thres_relerr
end
