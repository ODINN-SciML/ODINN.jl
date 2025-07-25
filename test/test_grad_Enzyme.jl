
function test_grad_Enzyme_SIAD2D()
    Random.seed!(1234)

    rgi_ids = ["RGI60-11.03638"]
    rgi_paths = get_rgi_paths()

    tspan = (2010.0, 2015.0)
    δt = 1/12
    params = Parameters(
        simulation = SimulationParameters(
            use_MB=false,
            velocities=true,
            tspan=tspan,
            step=δt,
            multiprocessing=false,
            workers=1,
            test_mode=true,
            rgi_paths=rgi_paths),
        UDE = UDEparameters(
            sensealg=SciMLSensitivity.ZygoteAdjoint(),
            optim_autoAD=ODINN.NoAD(),
            grad=DiscreteAdjoint(VJP_method=ODINN.EnzymeVJP()),
            optimization_method="AD+AD",
            target = :A),
        solver = Huginn.SolverParameters(
            step=δt,
            save_everystep=true,
            progress=true)
    )

    nn_model = NeuralNetwork(params)
    model = Model(
        iceflow = SIA2Dmodel(params; A=LawA(nn_model, params)),
        mass_balance = nothing,
        regressors = (; A=nn_model)
    )

    glaciers = initialize_glaciers(rgi_ids, params)

    glacier_idx = 1
    batch_idx = 1

    H = glaciers[glacier_idx].H₀

    simulation = FunctionalInversion(model, glaciers, params)
    simulation.cache = init_cache(model, simulation, glacier_idx, params)

    t = tspan[1]
    θ = simulation.model.machine_learning.θ
    simulation.model.iceflow[batch_idx].glacier_idx = glacier_idx

    vecBackwardSIA2D = randn(size(H,1), size(H,2))
    vecBackwardSIA2D_enzyme_H = deepcopy(vecBackwardSIA2D)
    vecBackwardSIA2D_enzyme_θ = deepcopy(vecBackwardSIA2D)

    s = ODINN.generate_simulation_batches(simulation)[1]


    # dH = zero(H)
    # ODINN.SIA2D_adjoint!(θ, dH, H, s, smodel, t, batch_idx)

    @assert false "TODO: replace by SIA2D with laws"
    # dH = Huginn.SIA2D(H, simulation, t)


    dH_H = Enzyme.make_zero(H)
    ∂H_enzyme = Enzyme.make_zero(H)
    _simulation = Enzyme.make_zero(s)
    smodel = StatefulLuxLayer{true}(s.model.machine_learning.architecture, θ.θ, s.model.machine_learning.st)
    Enzyme.autodiff(
        Reverse, ODINN.SIA2D_adjoint!, Const,
        Enzyme.Const(θ),
        Duplicated(dH_H, vecBackwardSIA2D_enzyme_H),
        Duplicated(H, ∂H_enzyme),
        Enzyme.Duplicated(s, _simulation),
        Enzyme.Const(smodel),
        Enzyme.Const(t),
        Enzyme.Const(glacier_idx)
    )


    ∂θ_enzyme = Enzyme.make_zero(θ)
    _simulation = Enzyme.make_zero(s)
    _smodel = Enzyme.make_zero(smodel)
    _H = Enzyme.make_zero(H)
    dH_λ = Enzyme.make_zero(H)
    Enzyme.autodiff(
        Reverse, ODINN.SIA2D_adjoint!, Const,
        Duplicated(θ, ∂θ_enzyme),
        Duplicated(dH_λ, vecBackwardSIA2D_enzyme_θ),
        Duplicated(H, _H),
        Duplicated(s, _simulation),
        Duplicated(smodel, _smodel),
        Const(t),
        Const(glacier_idx)
    )

    ∂H, ∂θ = ODINN.VJP_λ_∂SIA_discrete(vecBackwardSIA2D, H, θ, simulation, t; batch_id=batch_idx)

    ratio_H, angle_H, relerr_H = stats_err_arrays(∂H, ∂H_enzyme)
    thres_ratio = 1e-10
    thres_angle = 1e-10
    thres_relerr = 2e-5
    if printDebug | !( (abs(ratio_H)<thres_ratio) & (abs(angle_H)<thres_angle) & (abs(relerr_H)<thres_relerr) )
        println("Gradient wrt H")
        printVecScientific("ratio  = ",[ratio_H],thres_ratio)
        printVecScientific("angle  = ",[angle_H],thres_angle)
        printVecScientific("relerr = ",[relerr_H],thres_relerr)
    end
    @test (abs(ratio_H)<thres_ratio) & (abs(angle_H)<thres_angle) & (abs(relerr_H)<thres_relerr)

    ratio_θ, angle_θ, relerr_θ = stats_err_arrays(∂θ, ∂θ_enzyme)
    thres_ratio = 1e-14
    thres_angle = 1e-15
    thres_relerr = 1e-14
    if printDebug | !( (abs(ratio_θ)<thres_ratio) & (abs(angle_θ)<thres_angle) & (abs(relerr_θ)<thres_relerr) )
        println("Gradient wrt θ")
        printVecScientific("ratio  = ",[ratio_θ],thres_ratio)
        printVecScientific("angle  = ",[angle_θ],thres_angle)
        printVecScientific("relerr = ",[relerr_θ],thres_relerr)
    end
    @test (abs(ratio_θ)<thres_ratio) & (abs(angle_θ)<thres_angle) & (abs(relerr_θ)<thres_relerr)

end
