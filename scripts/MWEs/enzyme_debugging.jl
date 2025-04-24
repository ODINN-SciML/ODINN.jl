import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise
using ODINN
using Enzyme
using Lux
using Test

Enzyme.API.strictAliasing!(false)

#=
function VJP_λ_∂SIA∂H(VJPMode::EnzymeVJP, λ, H, θ, simulation, t, batch_id)
    dH_H = Enzyme.make_zero(H)
    λ_∂f∂H = Enzyme.make_zero(H)
    _simulation = Enzyme.make_zero(simulation)
    smodel = StatefulLuxLayer{true}(simulation.model.machine_learning.architecture, θ.θ, simulation.model.machine_learning.st)

    λH = deepcopy(λ) # Need to copy because Enzyme changes the backward gradient in-place
    Enzyme.autodiff(
        Reverse, SIA2D_UDE!, Const,
        # Reverse, SIA2D_adjoint!, Const,
        Enzyme.Const(θ),
        Duplicated(dH_H, λH),
        Duplicated(H, λ_∂f∂H),
        Enzyme.Duplicated(simulation, _simulation),
        Enzyme.Const(smodel),
        Enzyme.Const(t),
        Enzyme.Const(batch_id)
    )
    return λ_∂f∂H, dH_H
end

function VJP_λ_∂SIA∂θ(VJPMode::EnzymeVJP, λ, H, θ, dH_H, dH_λ, simulation, t, batch_id)
    λ_∂f∂θ = Enzyme.make_zero(θ)
    _simulation = Enzyme.make_zero(simulation)
    smodel = StatefulLuxLayer{true}(simulation.model.machine_learning.architecture, θ.θ, simulation.model.machine_learning.st)
    _smodel = Enzyme.make_zero(smodel)
    _H = Enzyme.make_zero(H)

    λθ = deepcopy(λ) # Need to copy because Enzyme changes the backward gradient in-place
    Enzyme.autodiff(
        Reverse, SIA2D_UDE!, Const,
        # Reverse, SIA2D_adjoint!, Const,
        Duplicated(θ, λ_∂f∂θ),
        Duplicated(dH_λ, λθ),
        Duplicated(H, _H),
        Duplicated(simulation, _simulation),
        Duplicated(smodel, _smodel),
        Const(t),
        Const(batch_id)
    )
    # Run simple test that both closures are computing the same primal
    if !isnothing(dH_H)
        @assert dH_H ≈ dH_λ "Result from forward pass needs to coincide for both closures when computing the pullback."
    end
    return λ_∂f∂θ
end

function SIA2D_UDE!(_θ, _dH::Matrix{R}, _H::Matrix{R}, simulation::FunctionalInversion, smodel, t::R, batch_id::I) where {R <: Real, I <: Integer}

    # TODO: add assert statement that this is just when VJP is Enzyme

    # if isnothing(batch_id)
    #     ice_model = simulation.model.iceflow
    #     glacier = simulation.glaciers
    # else
    #     ice_model = simulation.model.iceflow[batch_id]
    #     glacier = simulation.glaciers[batch_id]
    # end

    # We load the ML model with the parameters
    smodel.ps = _θ.θ
    smodel.st = simulation.model.machine_learning.st

    # apply_parametrization! = simulation.model.machine_learning.target.apply_parametrization!
    simulation.model.machine_learning.target.apply_parametrization!(;
        H = _H, ∇S = nothing, θ = _θ,
        ice_model = simulation.model.iceflow[batch_id],
        ml_model = simulation.model.machine_learning,
        glacier = simulation.glaciers[batch_id],
        params = simulation.parameters
    )

    Huginn.SIA2D!(_dH, _H, simulation, t; batch_id = batch_id)

    return nothing
end

function apply_parametrization_target_A(; H, ∇S, θ, ice_model, ml_model, glacier, params)
    # function apply_parametrization_target_A(H, ∇S, θ, ice_model, ml_model, params, glacier) where {I <: Integer, SIM <: Simulation}
        # We load the ML model with the parameters
        nn_model = ml_model.architecture
        st = ml_model.st
        smodel = StatefulLuxLayer{true}(nn_model, θ.θ, st)
        min_NN = params.physical.minA
        max_NN = params.physical.maxA
        A = predict_A̅(smodel, [mean(glacier.climate.longterm_temps)], (min_NN, max_NN))[1]
        return A
    end
    
    function apply_parametrization_target_A!(; H, ∇S, θ, ice_model, ml_model, glacier, params)
        A = apply_parametrization_target_A(; H, ∇S, θ, ice_model, ml_model, glacier, params)
        ice_model.A[] = A
        ice_model.D = nothing
        ice_model.D_is_provided = false
        return nothing
    end

    =#


    ###############

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
            light=false, # for now we do the simulation like this (a better name would be dense)
            test_mode=true,
            rgi_paths=rgi_paths),
        UDE = UDEparameters(
            optim_autoAD=ODINN.NoAD(),
            grad=DiscreteAdjoint(VJP_method=ODINN.EnzymeVJP()),
            optimization_method="AD+AD",
            target = :A),
        solver = Huginn.SolverParameters(
            step=δt,
            save_everystep=true,
            progress=true)
    )

    model = Model(
        iceflow = SIA2Dmodel(params),
        mass_balance = nothing,
        machine_learning = NeuralNetwork(params)
    )

    glaciers = initialize_glaciers(rgi_ids, params)

    glacier_idx = 1
    batch_idx = 1

    simulation = FunctionalInversion(model, glaciers, params)

    initialize_iceflow_model!(model.iceflow[batch_idx], glacier_idx, glaciers[glacier_idx], params)

    t = tspan[1]
    θ = simulation.model.machine_learning.θ

    H = glaciers[glacier_idx].H₀
    _H = Enzyme.make_zero(H)
    ∇S = model.iceflow[batch_idx].∇S
    _θ = Enzyme.make_zero(θ)

    ml_model = simulation.model.machine_learning
    iceflow_model = simulation.model.iceflow[1]
    _iceflow_model = Enzyme.make_zero(iceflow_model)
    glacier = simulation.glaciers[1]
    params = simulation.parameters

    ODINN.apply_parametrization_target_A(; H=H, ∇S=∇S, θ=θ, iceflow_model=iceflow_model, ml_model=ml_model, glacier=glacier, params=params)

    function apply_parametrization_target_A_wrapper(H, ∇S, θ, iceflow_model, ml_model, glacier, params)
        return ODINN.apply_parametrization_target_A(;
            H=H, ∇S=∇S, θ=θ,
            iceflow_model=iceflow_model,
            ml_model=ml_model,
            glacier=glacier,
            params=params
        )
    end

    function apply_parametrization_target_A_wrapper!(H, ∇S, θ, iceflow_model, ml_model, glacier, params)
        return ODINN.apply_parametrization_target_A!(;
            H=H, ∇S=∇S, θ=θ,
            iceflow_model=iceflow_model,
            ml_model=ml_model,
            glacier=glacier,
            params=params
        )
    end

    apply_parametrization_target_A_wrapper(H, ∇S, θ, iceflow_model, ml_model, glacier, params)


    #### SIA2D!  ####

    function SIA2D!(
    dH::Matrix{R},
    H::Matrix{R},
    simulation::SIM,
    t::R,
    batch_id::I
) where {R <:Real, I <: Integer, SIM <: Simulation}

    # For simulations using Reverse Diff, an iceflow model per glacier is needed
    # if isnothing(batch_id)
    #     SIA2D_model = simulation.model.iceflow
    #     glacier = simulation.glaciers[SIA2D_model.glacier_idx[]]
    # else
    #     SIA2D_model = simulation.model.iceflow[batch_id] # We pick the right iceflow model for this glacier
    #     glacier = simulation.glaciers[batch_id]
    # end
    
    SIA2D_model = simulation.model.iceflow[batch_id]
    glacier = simulation.glaciers[batch_id]

    params = simulation.parameters
    H̄ = SIA2D_model.H̄
    A = SIA2D_model.A
    n = SIA2D_model.n
    B = glacier.B
    S = SIA2D_model.S
    dSdx = SIA2D_model.dSdx
    dSdy = SIA2D_model.dSdy
    D = SIA2D_model.D
    D_is_provided = SIA2D_model.D_is_provided
    Dx = SIA2D_model.Dx
    Dy = SIA2D_model.Dy
    dSdx_edges = SIA2D_model.dSdx_edges
    dSdy_edges = SIA2D_model.dSdy_edges
    ∇S = SIA2D_model.∇S
    ∇Sx = SIA2D_model.∇Sx
    ∇Sy = SIA2D_model.∇Sy
    Fx = SIA2D_model.Fx
    Fy = SIA2D_model.Fy
    Fxx = SIA2D_model.Fxx
    Fyy = SIA2D_model.Fyy
    Δx = glacier.Δx
    Δy = glacier.Δy
    Γ = SIA2D_model.Γ
    ρ = simulation.parameters.physical.ρ
    g = simulation.parameters.physical.g

    # First, enforce values to be positive
    map!(x -> ifelse(x > 0.0, x, 0.0), H, H)
    # Update glacier surface altimetry
    S .= B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    diff_x!(dSdx, S, Δx)
    diff_y!(dSdy, S, Δy)
    avg_y!(∇Sx, dSdx)
    avg_x!(∇Sy, dSdy)
    ∇S .= @. (∇Sx^2 + ∇Sy^2)^((n - 1) / 2)
    avg!(H̄, H)
    Γ .= @. 2.0 * A * (ρ * g)^n / (n + 2) # 1 / m^3 s
    D .= @. Γ * H̄^(n + 2) * ∇S

    # Compute flux components
    @views diff_x!(dSdx_edges, S[:,2:end - 1], Δx)
    @views diff_y!(dSdy_edges, S[2:end - 1,:], Δy)

    # # Cap surface elevaton differences with the upstream ice thickness to
    # # imporse boundary condition of the SIA equation
    # η₀ = params.physical.η₀
    # dSdx_edges .= @views @. min(dSdx_edges,  η₀ * H[2:end, 2:end-1] / Δx)
    # dSdx_edges .= @views @. max(dSdx_edges, -η₀ * H[1:end-1, 2:end-1] / Δx)
    # dSdy_edges .= @views @. min(dSdy_edges,  η₀ * H[2:end-1, 2:end] / Δy)
    # dSdy_edges .= @views @. max(dSdy_edges, -η₀ * H[2:end-1, 1:end-1] / Δy)

    avg_y!(Dx, D)
    avg_x!(Dy, D)
    # Fx .= .-Dx .* dSdx_edges
    # Fy .= .-Dy .* dSdy_edges

    # #  Flux divergence
    # diff_x!(Fxx, Fx, Δx)
    # diff_y!(Fyy, Fy, Δy)
    # inn(dH) .= .-(Fxx .+ Fyy)

    return nothing
end

    dH = deepcopy(H)
    _dH = Enzyme.make_zero(dH)
    # λ_∂f∂H = Enzyme.make_zero(H)
    _simulation = Enzyme.make_zero(simulation)
    # smodel = StatefulLuxLayer{true}(simulation.model.machine_learning.architecture, θ.θ, simulation.model.machine_learning.st)
    batch_id = 1

    function SIA2D!_wrapper(dH, H, simulation, t, batch_id)
        return Huginn.SIA2D!(dH, H, simulation, t; batch_id=batch_id)
    end
    # Huginn.SIA2D!(_dH, _H, simulation, t; batch_id = batch_id)

    Enzyme.autodiff(
        Reverse, SIA2D!, Const,
        Duplicated(dH, _dH),
        Duplicated(H, _H),
        Duplicated(simulation, _simulation),
        Const(t),
        Const(batch_id)
    )

    @code_warntype SIA2D!_wrapper(dH, H, simulation, t, batch_id)

    @code_warntype Huginn.SIA2D(H, simulation, t; batch_id = batch_id)

    function SIA2D_wrapper(H, simulation, t, batch_id)
        return Huginn.SIA2D(H, simulation, t; batch_id=batch_id)
    end

    Enzyme.autodiff(
        Reverse, SIA2D_wrapper, Duplicated,
        Const(H),
        Duplicated(simulation, _simulation),
        Const(t),
        Const(batch_id)
    )

    Enzyme.autodiff(
        Reverse, SIA2D!_wrapper, Const,
        Duplicated(dH, _dH),
        Duplicated(H, _H),
        Duplicated(simulation, _simulation),
        Const(t),
        Const(batch_id)
    )


    #### APPLY_PARAMETRIZATION_TARGET_A  ####

    # Enzyme.autodiff(
    #     Reverse, apply_parametrization_target_A_wrapper!, 
    #     Duplicated(H, _H),
    #     Const(∇S),
    #     Duplicated(θ, _θ),
    #     Duplicated(iceflow_model, _iceflow_model),
    #     Const(ml_model),
    #     Const(glacier),
    #     Const(params),
    # )

    # @show _θ
    # @show _H


    #### SIA2D_UDE! ####S

    # dH_H = Enzyme.make_zero(H)
    # λ_∂f∂H = Enzyme.make_zero(H)
    # _simulation = Enzyme.make_zero(simulation)
    # smodel = StatefulLuxLayer{true}(simulation.model.machine_learning.architecture, θ.θ, simulation.model.machine_learning.st)
    # batch_id = 1

    # λH = deepcopy(H) # Need to copy because Enzyme changes the backward gradient in-place
    # Enzyme.autodiff(
    #     Reverse, ODINN.SIA2D_UDE!, Const,
    #     # Reverse, SIA2D_adjoint!, Const,
    #     Enzyme.Const(θ),
    #     Duplicated(dH_H, λH),
    #     Duplicated(H, λ_∂f∂H),
    #     Enzyme.Duplicated(simulation, _simulation),
    #     Enzyme.Const(smodel),
    #     Enzyme.Const(t),
    #     Enzyme.Const(batch_id)
    # )

