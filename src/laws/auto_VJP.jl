import Sleipnir: prepare_vjp_law, ∂law∂θ!, ∂law∂inp!

"""
    struct VJPsPrepLaw <: AbstractPrepVJP

A container struct that holds all objects needed to compute vector-Jacobian
products (VJPs) for a law using DifferentiationInterface.

Fields:
- `f_θ_first`: Function to evaluate the law with parameters θ as the first argument.
- `f_inp_first`: Function to evaluate the law with inputs as the first argument.
- `prep_θ`: Precomputed gradient preparation for parameters θ.
- `prep_inp`: Precomputed gradient preparation for inputs.

This struct is used to prepare the VJP computation with DifferentiationInterface (DI).
Depending on the AD backend, DI might require to precompile code and this struct stores the results.
This allows each VJP call to be fast in the adjoint PDE by reusing the preparation results.
"""
mutable struct VJPsPrepLaw <: AbstractPrepVJP
    f_θ_first
    f_inp_first
    prep_θ
    prep_inp
end

prepare_vjp_law(
    simulation::Inversion,
    law::Union{Law{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Sleipnir.GenInputsAndApply{<:Any, <:Function}, CustomVJP}, Law{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Function, CustomVJP}},
    law_cache,
    θ,
    glacier_idx,
) = nothing
function prepare_vjp_law(
    simulation::Inversion,
    law::Union{
        Law{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, DIVJP},
        Law{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Sleipnir.GenInputsAndApply{<:Any, DIVJP}, <:Any},
    },
    law_cache,
    θ,
    glacier_idx,
)
    # Don't prepare the VJPs with DI if it's not necessary
    if isa(simulation.parameters.UDE.grad, SciMLSensitivityAdjoint) || isa(simulation.parameters.UDE.grad, DummyAdjoint)
        return nothing
    end
    if isa(simulation.parameters.UDE.grad.VJP_method, EnzymeVJP)
        return nothing
    end

    # Define wrapper function that depends only on θ and inp
    f_θ_first = let law=law, law_cache=law_cache
        # We don't use apply_law! because we want to evaluate with custom inputs
        function (_θ, _inp)
            law.f.f(law_cache, _inp, _θ)
        end
    end
    # It seems that with DI when Constant is being used, it must be at the last position
    f_inp_first(θ, inp) = f_θ_first(inp, θ)

    # Initialize inputs with zeros
    values_inputs = map(law.f.inputs) do input
        zero(input, simulation, glacier_idx)
    end

    # Transform inputs to scalar inputs
    packed_inputs = map(values_inputs) do val
        only(similar(val,()))
    end

    backend = simulation.parameters.UDE.grad.VJP_method.regressorADBackend
    prep_θ = DI.prepare_gradient(f_θ_first, backend, θ, DI.Constant(packed_inputs))
    prep_inp = DI.prepare_gradient(f_inp_first, backend, packed_inputs, DI.Constant(θ))
    return VJPsPrepLaw(f_θ_first, f_inp_first, prep_θ, prep_inp)
end

function ∂law∂inp!(backend::DI.AutoZygote, vjpsPrepLaw::AbstractPrepVJP, inp, θ)
    tmp = DI.gradient(vjpsPrepLaw.f_inp_first, vjpsPrepLaw.prep_inp, backend, inp, DI.Constant(θ))
    values(tmp)
end
function ∂law∂θ!(backend::DI.AutoZygote, vjpsPrepLaw::AbstractPrepVJP, inp, θ)
    DI.gradient(vjpsPrepLaw.f_θ_first, vjpsPrepLaw.prep_θ, backend, θ, DI.Constant(inp))
end

function ∂law∂inp!(backend::DI.AutoMooncake, vjpsPrepLaw::AbstractPrepVJP, inp, θ)
    tmp = DI.gradient(vjpsPrepLaw.f_inp_first, vjpsPrepLaw.prep_inp, backend, inp, DI.Constant(θ)) #.fields.data
    values(tmp)
end
function ∂law∂θ!(backend::DI.AutoMooncake, vjpsPrepLaw::AbstractPrepVJP, inp, θ)
    DI.gradient(vjpsPrepLaw.f_θ_first, vjpsPrepLaw.prep_θ, backend, θ, DI.Constant(inp)).fields.data
end

function ∂law∂inp!(backend, law::Law{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, CustomVJP}, law_cache, vjpsPrepLaw, inp, θ)
    law.f_VJP_input.f(law_cache, inp, θ)
end
function ∂law∂inp!(backend, law::Law{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, DIVJP}, law_cache, vjpsPrepLaw::AbstractPrepVJP, inp, θ)
    law_cache.vjp_inp .= ∂law∂inp!(backend, vjpsPrepLaw, inp, θ)
end

function ∂law∂θ!(backend, law::Law{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, CustomVJP}, law_cache, vjpsPrepLaw, inp, θ)
    law.f_VJP_θ.f(law_cache, inp, θ)
end
function ∂law∂θ!(backend, law::Law{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, DIVJP}, law_cache, vjpsPrepLaw::AbstractPrepVJP, inp, θ)
    law_cache.vjp_θ .= ∂law∂θ!(backend, vjpsPrepLaw, inp, θ)
end
