
function VJP_λ_∂SIA∂H(VJPMode::DiscreteVJP, λ, H, θ, simulation::Simulation, t)
    λ_∂f∂H = VJP_λ_∂SIA∂H_discrete(λ, H, θ, simulation, t)
    return λ_∂f∂H, nothing
end

function VJP_λ_∂SIA∂H(VJPMode::ContinuousVJP, λ, H, θ, simulation::Simulation, t)
    λ_∂f∂H = VJP_λ_∂SIA∂H_continuous(λ, H, θ, simulation, t)
    return λ_∂f∂H, nothing
end

function VJP_λ_∂SIA∂H(VJPMode::EnzymeVJP, λ, H, θ, simulation::Simulation, t)
    dH_H = Enzyme.make_zero(H)
    λ_∂f∂H = Enzyme.make_zero(H)
    _simulation = Enzyme.make_zero(simulation)
    _θ = Enzyme.make_zero(θ)

    λH = deepcopy(λ) # Need to copy because Enzyme changes the backward gradient in-place
    Enzyme.autodiff(
        EnzymeCore.Reverse, SIA2D_UDE!, Const,
        Duplicated(θ, _θ),
        Duplicated(dH_H, λH),
        Duplicated(H, λ_∂f∂H),
        Duplicated(simulation, _simulation),
        Const(t)
    )
    return λ_∂f∂H, dH_H
end

function VJP_λ_∂SIA∂θ(VJPMode::DiscreteVJP, λ, H, θ, dH_H, simulation::Simulation, t)
    λ_∂f∂θ = VJP_λ_∂SIA∂θ_discrete(λ, H, θ, simulation, t)
    return λ_∂f∂θ
end

function VJP_λ_∂SIA∂θ(VJPMode::ContinuousVJP, λ, H, θ, dH_H, simulation::Simulation, t)
    λ_∂f∂θ = VJP_λ_∂SIA∂θ_continuous(λ, H, θ, simulation, t)
    return λ_∂f∂θ
end

function VJP_λ_∂SIA∂θ(VJPMode::EnzymeVJP, λ, H, θ, dH_H, simulation::Simulation, t)
    λ_∂f∂θ = Enzyme.make_zero(θ)
    _simulation = Enzyme.make_zero(simulation)

    dH_λ = Enzyme.make_zero(H)
    λθ = deepcopy(λ) # Need to copy because Enzyme changes the backward gradient in-place
    Enzyme.autodiff(
        EnzymeCore.Reverse, SIA2D_UDE!, Const,
        Duplicated(θ, λ_∂f∂θ),
        Duplicated(dH_λ, λθ),
        Const(H),
        Duplicated(simulation, _simulation),
        Const(t)
    )
    # Run simple test that both closures are computing the same primal
    if !isnothing(dH_H)
        @assert dH_H ≈ dH_λ "Result from forward pass needs to coincide for both closures when computing the pullback."
    end
    return λ_∂f∂θ
end

function VJP_λ_∂surface_V∂H(VJPMode::DiscreteVJP, λx, λy, H, θ, simulation, t)
    λ_∂V∂H = VJP_λ_∂surface_V∂H_discrete(λx, λy, H, θ, simulation, t)
    return λ_∂V∂H, nothing
end

function VJP_λ_∂surface_V∂θ(VJPMode::DiscreteVJP, λx, λy, H, θ, simulation, t)
    λ_∂V∂H = VJP_λ_∂surface_V∂θ_discrete(λx, λy, H, θ, simulation, t)
    return λ_∂V∂H, nothing
end

function MB_wrapper!(MB, H, simulation, glacier, step)
    model = simulation.model
    cache = simulation.cache
    glacier.S .= glacier.B .+ H

    # Below we call the functions that are inside MB_timestep! manually
    # This is because get_cumulative_climate! cannot be differentiated with Enzyme, so it is called beforehand in the VJP function to retrieve the cumulative climate
    downscale_2D_climate!(glacier)
    cache.iceflow.MB .= compute_MB(model.mass_balance, glacier.climate.climate_2D_step, step)

    apply_MB_mask!(H, cache.iceflow)
    MB .= simulation.cache.iceflow.MB
end
function VJP_λ_∂MB∂H(VJPMode::EnzymeVJP, λ, H, simulation::Simulation, glacier, t)
    step_MB = simulation.parameters.simulation.step_MB
    # Differentiation of get_cumulative_climate! with Enzyme yields an error
    # Since it isn't involved in the gradient computation (doesn't depend on H), it can be computed beforehand
    get_cumulative_climate!(glacier.climate, t, step_MB)

    _simulation = Enzyme.make_zero(simulation)
    _glacier = Enzyme.make_zero(glacier)
    _H = deepcopy(H) # Copy H since it is modified in-place
    λ_∂MB∂H = Enzyme.make_zero(H)
    MB = Enzyme.make_zero(H)
    λH = deepcopy(λ) # Need to copy because Enzyme changes the backward gradient in-place
    Enzyme.autodiff(
        EnzymeCore.Reverse, MB_wrapper!, Const,
        Duplicated(MB, λH),
        Duplicated(_H, λ_∂MB∂H),
        Duplicated(simulation, _simulation),
        Duplicated(glacier, _glacier),
        Const(step_MB)
    )
    return λ_∂MB∂H
end

function VJP_λ_∂MB∂H(VJPMode::DiscreteVJP, λ, H, simulation::Simulation, glacier, t)
    model = simulation.model
    cache = simulation.cache
    step_MB = simulation.parameters.simulation.step_MB
    glacier.S .= glacier.B .+ H
    get_cumulative_climate!(glacier.climate, t, step_MB)

    mb_model = simulation.model.mass_balance
    λ_∂MB∂H = if isa(mb_model, TImodel1)
        downscale_2D_climate!(glacier)
        climate_2D_step = glacier.climate.climate_2D_step

        PDD = glacier.climate.climate_step.temp .+
              climate_2D_step.gradient .* (glacier.S .- climate_2D_step.ref_hgt)
        PDD_jac = climate_2D_step.gradient .* λ
        PDD_jac .= ifelse.(PDD .< 0.0, 0.0, PDD_jac)

        cache.iceflow.MB .= compute_MB(model.mass_balance, glacier.climate.climate_2D_step, step_MB)
        MB = cache.iceflow.MB
        MB_mask = cache.iceflow.MB_mask

        # MB, MB_mask, MB_total = ifm.MB, ifm.MB_mask, ifm.MB_total
        MB_mask .= ((H .> 0.0) .&& (MB .< 0.0)) .|| ((H .> 10.0) .&& (MB .>= 0.0))
        # Set MB to zero outside of MB_mask
        MB[.!MB_mask] .= 0
        # Get the linear indices where MB_mask is true
        mask_indices = findall(MB_mask)
        # Among those, find where ice would disappear after MB application
        mask_ice_disappear = (H[mask_indices] .+ MB[mask_indices]) .< 0.0
        # Get the actual indices to modify
        disappear_indices = mask_indices[mask_ice_disappear]
        # Clip MB in-place at those indices
        MB[disappear_indices] .= .-H[disappear_indices]

        λ_∂MB∂H = zero(λ)
        # The snow term doesn't depend on the ice thickness, hence it is null
        λ_∂MB∂H[MB_mask] = ((.- (mb_model.DDF .* PDD_jac)) ./ (step_MB / (1 / 12)))[MB_mask]
        λ_∂MB∂H[disappear_indices] .= -λ[disappear_indices]
        λ_∂MB∂H
    else
        throw("The discrete VJP for model $(typeof(mb_model)) is not supported yet.")
    end

    return λ_∂MB∂H
end

function VJP_λ_∂MB∂H(VJPMode::NoVJP, λ, H, simulation::Simulation, glacier, t)
    return zero(λ)
end

# ============================================================================
# Automatic (SciMLSensitivityAdjoint) differentiation of the H→V map.
# The post-solve velocity computation `V_from_H`/`surface_V` mutates the law
# cache, so Zygote cannot differentiate it. We give `V_from_H` an rrule whose
# pullback computes the VJP with Enzyme (which handles the in-place cache and
# laws), mirroring `VJP_λ_∂SIA∂H(::EnzymeVJP, …)`. This unblocks LossV/LossHV/
# VelocityRegularization and time-aggregated velocity losses for any iceflow
# model and regressor, without hand-written per-loss adjoints.
# ============================================================================

# Enzyme VJP of the H→(Vx,Vy) map: returns (∂H, ∂θ) for seeds (dVx,dVy).
# dVx and dVy must already include any ΔV folded analytically by the caller (the
# V=‖(Vx,Vy)‖ magnitude is handled there, so Enzyme never sees the sqrt at V=0).
# Differentiates `Huginn.surface_V_inplace!` directly: it writes Vx,Vy into the
# passed H-sized buffers and returns nothing, mirroring the proven `SIA2D!` Enzyme
# target (the tuple-returning `surface_V!` trips Enzyme's verifier). V_from_H also
# returns H-sized Vx,Vy, so the cotangents dVx,dVy match the H-sized primal buffers.
function ∂V_from_H_vjp!(dVx, dVy, H, θ, simulation, t)
    _Vx = zero(H)
    _Vy = zero(H)
    dH = Enzyme.make_zero(H)
    dθ = Enzyme.make_zero(θ)
    _sim = Enzyme.make_zero(simulation)
    # Runtime activity needed for deterministic compilation (matches SciML's EnzymeVJP).
    Enzyme.autodiff(
        EnzymeCore.set_runtime_activity(EnzymeCore.Reverse), Huginn.surface_V_inplace!, Const,
        Duplicated(_Vx, dVx),
        Duplicated(_Vy, dVy),
        Duplicated(H, dH),
        Duplicated(simulation, _sim),
        Const(t),
        Duplicated(θ, dθ)
    )
    return dH, dθ
end

# Lets Zygote (SciMLSensitivityAdjoint path) differentiate velocity-based losses.
function ChainRules.rrule(::typeof(Huginn.V_from_H), simulation::Inversion, H, t, θ)
    Vx, Vy, V = Huginn.V_from_H(simulation, H, t, θ)
    function V_from_H_pullback(Δ)
        if Δ isa ChainRules.AbstractZero
            return (ChainRules.NoTangent(), ChainRules.NoTangent(),
                ChainRules.ZeroTangent(), ChainRules.NoTangent(), ChainRules.ZeroTangent())
        end
        # Fold all three output cotangents into (dVx, dVy) seeds for Enzyme.
        # ΔV through V=‖(Vx,Vy)‖ is handled analytically here to avoid 0/0 NaN
        # at ice-free cells where Vx=Vy=V=0.
        dVx = Δ[1] isa ChainRules.AbstractZero ? zero(Vx) : copy(ChainRules.unthunk(Δ[1]))
        dVy = Δ[2] isa ChainRules.AbstractZero ? zero(Vy) : copy(ChainRules.unthunk(Δ[2]))
        if !(Δ[3] isa ChainRules.AbstractZero)
            dV = ChainRules.unthunk(Δ[3])
            # ∂V/∂Vx = Vx/V; zero where V=0 (Vx=0 there, so cotangent is 0)
            @. dVx += ifelse(V > 0, dV * Vx / V, zero(eltype(Vx)))
            @. dVy += ifelse(V > 0, dV * Vy / V, zero(eltype(Vy)))
        end
        ∂H, ∂θ = ∂V_from_H_vjp!(dVx, dVy, H, θ, simulation, t)
        return (ChainRules.NoTangent(), ChainRules.NoTangent(), ∂H,
            ChainRules.NoTangent(), ∂θ)
    end
    return (Vx, Vy, V), V_from_H_pullback
end
