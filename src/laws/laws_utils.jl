export eval_law

"""
    eval_law(law::AbstractLaw, simulation::Simulation, glacier_idx::Integer, inputs::NamedTuple, θ)

Evaluates a law on the specified glacier within a simulation context and for a user defined input.

# Arguments
- `law::AbstractLaw`: The law object to be evaluated. Must provide a function `f` and an `init_cache` method.
- `simulation::Simulation`: The simulation context, containing model parameters and machine learning components.
- `glacier_idx::Integer`: Index identifying which glacier in the simulation to evaluate the law for.
- `input_values::NamedTuple`: Input data required by the law and provided by the user.
- `θ`: Weights used in the law to make inference. This can be `nothing` when the law has no parameter.

# Returns
- The updated cache after evaluating the law. The cache contains the result of the law's computation for the specified glacier and inputs.

# Details
- The function initializes a cache for the law using `init_cache`.
- If the simulation has a machine learning model, the model's parameters (`θ`) are updated in-place with the provided `θ`.
- The law's function is then called with the cache, inputs, and parameters. The result is stored in the cache and the cache is returned.
- In future versions, the design may change so that only `inputs` and `θ` are needed, with the cache handled separately so that no `simulation` is required.

# Example
```julia
result = eval_law(simulation.model.iceflow.A, simulation, glacier_idx, (; T=273.15), θ)
````
"""
function eval_law(law::AbstractLaw, simulation::Simulation, glacier_idx::Integer, input_values::NamedTuple, θ)
    # Initialize the cache to be able to make an inference of the law
    params = simulation.parameters

    cache = init_cache(law, simulation, glacier_idx, params)
    if !isnothing(simulation.model.machine_learning)
        simulation.model.machine_learning.θ = θ
    end

    law.f.f(cache, input_values, θ)
    return cache.value
end


"""
    T_A_Alaw(simulation::Simulation, glacier_idx::Integer, θ, t::AbstractFloat)

Evaluate the A law when it defines a mapping between the long term air temperature and the creep coefficient `A` and return both the input temperature `T` and the computed creep coefficient `A`.

# Arguments
- `simulation::Simulation`: The simulation object containing model data and parameters.
- `glacier_idx::Integer`: Index specifying which glacier to evaluate.
- `θ`: Model parameters to be used in the law.
- `t::AbstractFloat`: The time at which to evaluate the law. For this law it is useless
    but in the general setting, a law needs a time `t` in order to retrieve the inputs.
    For the sake of consistency, this input was kept.

# Returns
- `(T, A)`: A tuple containing:
    - `T`: The input long term air temperature for the specified glacier.
    - `A`: The evaluated creep coefficient for the specified glacier.

# Details
- The function checks that the inputs of the A law are exactly as expected (long term air temperature only).
- Retrieves the long term air temperature for the specific glacier.
- Evaluates the creep coefficient using the law.
- Returns both the temperature and creep coefficient as a tuple. Since the cache of `A` is a zero dimensional array, it is converted to float before returning the value.

# Example
```julia
T, A = T_A_Alaw(simulation, glacier_idx, θ, 2010.0)
"""
function T_A_Alaw(simulation::Simulation, glacier_idx::Integer, θ, t::AbstractFloat)
    _inputs_A_law = (; T=iTemp())
    @assert inputs(simulation.model.iceflow.A)==_inputs_A_law "The function T_A_Alaw can be called only when the inputs of the A law are $(_inputs_A_law)."

    T = get_input(iTemp(), simulation, glacier_idx, t)
    A = eval_law(simulation.model.iceflow.A, simulation, glacier_idx, (;T=T), θ)

    return T, A[]
end
