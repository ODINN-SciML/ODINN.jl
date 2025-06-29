export eval_law

# Currently we need to provide simulation to `eval_law` but in the future we plan to handle this with a separate cache that will allow us to provide only the inputs and θ when evaluating the law
function eval_law(law::AbstractLaw, simulation::Simulation, glacier_idx::Integer, inputs, θ)
    # Initialize the cache to be able to make an inference of the law
    params = simulation.parameters
    cache = init_cache(law, simulation, glacier_idx, params)
    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ .= θ
    end

    law.f.f(cache, inputs, θ)
    return cache
end


function T_A_Alaw(simulation::Simulation, glacier_idx::Integer, θ, t::AbstractFloat)
    _inputs_A_law = (; T=InpTemp())
    @assert inputs(simulation.model.iceflow.A)==_inputs_A_law "The function T_A_Alaw can be called only when the inputs of the A law are $(_inputs_A_law)."

    T = get_input(InpTemp(), simulation, glacier_idx, t)
    A = eval_law(simulation.model.iceflow.A, simulation, glacier_idx, (;T=T), θ)

    return T, A[]
end
