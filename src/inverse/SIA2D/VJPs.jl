
function VJP_λ_∂SIA∂H(VJPMode::DiscreteVJP, λ, H, θ, simulation, t, batch_id)
    λ_∂f∂H = VJP_λ_∂SIA_discrete(λ, H, θ, simulation, t; batch_id = batch_id)[1]
    return λ_∂f∂H, nothing
end

function VJP_λ_∂SIA∂H(VJPMode::ContinuousVJP, λ, H, θ, simulation, t, batch_id)
    λ_∂f∂H = VJP_λ_∂SIA∂H_continuous(λ, H, θ, simulation, t; batch_id = batch_id)
    return λ_∂f∂H, nothing
end

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

function VJP_λ_∂SIA∂θ(VJPMode::DiscreteVJP, λ, H, θ, dH_H, dH_λ, simulation, t, batch_id)
    λ_∂f∂θ = VJP_λ_∂SIA_discrete(λ, H, θ, simulation, t; batch_id = batch_id)[2]
    return λ_∂f∂θ
end

function VJP_λ_∂SIA∂θ(VJPMode::ContinuousVJP, λ, H, θ, dH_H, dH_λ, simulation, t, batch_id)
    λ_∂f∂θ = VJP_λ_∂SIA∂θ_continuous(λ, H, θ, simulation, t; batch_id = batch_id)
    return λ_∂f∂θ
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
