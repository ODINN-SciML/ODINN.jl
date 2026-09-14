# # Laws inputs

# This tutorial showcases the different inputs that can be used inside the laws.

using ODINN

# If we represent the A law that we already presented in the [Laws](./laws.md) tutorial, we can see that it depends on an input `T`, which is the long term air temperature:

params = Parameters() # Dummy parameters
nn_model = NeuralNetwork(params)
A_law = LawA(nn_model, params)

# The inputs of the laws are represented as objects that derive from `AbstractInput`.
# We have defined a subset of inputs that can be used to parameterize laws.

# ## Implementation

# It is also possible to define new inputs by creating a new struct type and defining the method for this specific type.
# For example the long term air temperature is defined with the following code:

# ```julia
# struct iTemp <: AbstractInput end
# default_name(::iTemp) = :long_term_temperature
# function get_input(::iTemp, simulation, glacier_idx, t)
#     glacier = simulation.glaciers[glacier_idx]
#     return mean(glacier.climate.longterm_temps)
# end
# function Base.zero(::iTemp, simulation, glacier_idx)
#     glacier = simulation.glaciers[glacier_idx]
#     return zero(glacier.climate.longterm_temps)
# end
# ```

# An input can compute a physical quantity which is not already defined in the iceflow model like the long term air temperature above, but it can also re-use existing variables.
# This is the case of `∇S` which corresponds to the surface slope.
# It is retrieved simply by returning the cached variable that lives in the iceflow model:
# ```julia
# struct i∇S <: AbstractInput end
# default_name(::i∇S) = :∇S
# function get_input(::i∇S, simulation, glacier_idx, t)
#     return simulation.cache.iceflow.∇S
# end
# function Base.zero(::i∇S, simulation, glacier_idx)
#     (; nx, ny) = simulation.glaciers[glacier_idx]
#     return zeros(nx-1, ny-1)
# end
# ```

# !!! warning
#     Keep in mind that if you define new inputs, this will work only for forward simulations. As soon as you do inversions with custom gradient computation (that is not with SciMLSensitivity), this requires a careful rooting of the gradient and you need to implement more than simply defining new inputs that depend on the glacier state. This is considered as a very advanced feature and we recommend that you seek assistance in this case.

# ## List of available inputs

# For the moment we support the following list of inputs:

# - Long term air temperature

iTemp()

# - Cumulative positive degree days (PDD)

iCPDD()

# - Ice thickness on the dual grid in the SIA

# This is the variable to use for the ice thickness.

iH̄()

# - Surface slope

i∇S()

# - Topographic roughness

# It can be the roughness of the bed or of the surface depending on the selected options.
# There are several ways to compute this quantity which result in different properties.

iTopoRough()