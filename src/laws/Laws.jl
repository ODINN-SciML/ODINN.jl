import Sleipnir: get_input, default_name

export LawA, LawY, LawU

"""
    _pred_NN(inp::Vector{F}, smodel, θ, prescale, postscale) where {F <: AbstractFloat}

Compute the output of a neural network model on the input vector `inp`.

# Arguments
- `inp::Vector{F}`: Input vector of floats.
- `smodel`: The neural network model.
- `θ`: Parameters for the neural network model.
- `prescale`: Function to scale the input vector before passing it to the model.
- `postscale`: Function to scale the model output.

# Returns
- The single (scalar) output value from the neural network after applying `prescale`
    to the input, evaluating the model, and then applying `postscale`. The result is
    extracted via `only`.

# Notes
- The function assumes that the neural network, when evaluated, returns an iterable with exactly one element.
- Using `only` will throw an error if the output is not exactly one element.

# Example
```julia
mymodel = StatefulLuxLayer{true}(archi, nothing, st)
y = _pred_NN([1.0, 2.0], mymodel, θ, prescale_fn, postscale_fn)
"""
function _pred_NN(inp::Vector{F}, smodel, θ, prescale, postscale) where {F <: AbstractFloat}
    return only(postscale(smodel(prescale(inp), θ)))
end

"""
    LawU(
        nn_model::NeuralNetwork,
        params::Sleipnir.Parameters;
        max_NN::Union{F, Nothing} = 50.0,
        prescale_bounds::Union{Vector{Tuple{F,F}}, Nothing} = [(0.0, 300.0), (0.0, 0.5)],
    ) where {F <: AbstractFloat}

Constructs a law object for the diffusive velocity `U` in the SIA based on a neural
network that takes as input the ice thickness `H̄` and the surface slope `∇S`.
The diffusive velocity `U` with this law is a matrix and the diffusivity in the SIA
is obtained through D = U * H̄.
See also `SIA2D_D_target`.

# Arguments
- `nn_model::NeuralNetwork`: A neural network model containing the architecture
    `archi` and state `st` used for evaluation of the law.
- `params::Sleipnir.Parameters`: Parameters struct. Not used for the moment but kept
    as an argument to keep consistency with other equivalent functions `LawA` and
    `LawY`.
- `max_NN::Union{F, Nothing}`: Expected maximum value of the neural network output.
    If set to `nothing`, no postscaling is applied.
- `prescale_bounds::Union{Vector{Tuple{F,F}}, Nothing}`: Vector of tuples where each
    tuple defines the lower and upper bounds of the input for scaling.
    If set to `nothing`, no prescaling is applied.

# Returns
- `U_law`: A `Law{Array{Float64, 2}}` instance that computes the diffusive velocity `U`
    based on the ice thickness `H̄` and the surface slope `∇S` using the neural network.
    The law scales the output using the `max_NN` argument.

# Notes
- The computation is compatible with Zygote for automatic differentiation.

# Details
- The function wraps the architecture and state of the neural network in a`StatefulLuxLayer`.
- The resulting law takes input variables, applies the neural network, and scales its output
    to match `max_NN`.
- The in-place assignment to `cache` is ignored in differentiation to allow gradient
    computation with Zygote.
- The `init_cache` function initializes the cache with a zero matrix.

# Example
```julia
nn_model = NeuralNetwork(params)
bounds_H = (0.0, 300.0)
bounds_∇S = (0.0, 0.5)
U_law = LawU(nn_model, params; max_NN = 50.0, prescale_bounds = [bounds_H, bounds_∇S])
"""
function LawU(
    nn_model::NeuralNetwork,
    params::Sleipnir.Parameters;
    max_NN::Union{F, Nothing} = 50.0,
    prescale_bounds::Union{Vector{Tuple{F,F}}, Nothing} = [(0.0, 300.0), (0.0, 0.5)],
) where {F <: AbstractFloat}
    prescale = !isnothing(prescale_bounds) ? X -> _ml_model_prescale(X, prescale_bounds) : identity
    # The value of max_NN should correspond to maximum of Umax * dSdx
    postscale = !isnothing(max_NN) ? Y -> _ml_model_postscale(Y, max_NN) : identity

    archi = nn_model.architecture
    st = nn_model.st
    smodel = StatefulLuxLayer{true}(archi, nothing, st)

    U_law = let smodel = smodel, prescale = prescale, postscale = postscale
    Law{Array{Float64, 2}}(;
        inputs = (; H̄=iH̄(), ∇S=i∇S()),
        f! = function (cache, inp, θ)
            D = ((h, ∇s) -> _pred_NN([h, ∇s], smodel, θ.U, prescale, postscale)).(inp.H̄, inp.∇S)

            # Flag the in-place assignment as non differented and return D instead in
            # order to be able to compute ∂D∂θ with Zygote
            Zygote.@ignore cache .= D
            return D
        end,
        init_cache = function (simulation, glacier_idx, θ; scalar::Bool = true)
            (; nx, ny) = simulation.glaciers[glacier_idx]
            return zeros(nx-1, ny-1)
        end,
    )
    end
    return U_law
end


"""

    LawY(
        nn_model::NeuralNetwork,
        params::Sleipnir.Parameters;
        max_NN::Union{F, Nothing} = nothing,
        prescale_bounds::Vector{Tuple{F,F}} = [(-25.0, 0.0), (0.0, 500.0)],
    ) where {F <: AbstractFloat}

Constructs a law object for the hybrid diffusivity `Y` in the SIA based on a neural
network that takes as input the long term air temperature and the ice thickness `H̄`.
The hybrid diffusivity `Y` with this law is a matrix as it depends on the ice thickness.
This law is used in an hybrid setting where the `n` exponent in the mathematical
expression of the diffusivity is different from the one used to generate the ground
truth. The goal of this law is to retrieve the missing part of the diffusivity.
Please refer to `SIA2D_D_hybrid_target` for a mathematical definition.

# Arguments
- `nn_model::NeuralNetwork`: A neural network model containing the architecture
    `archi` and state `st` used for evaluation of the law.
- `params::Sleipnir.Parameters`: Parameters struct used to retrieve the maximum
    value of A for scaling of the neural network output.
- `max_NN::Union{F, Nothing}`: Expected maximum value of the neural network output.
    If not specified, the law takes as an expected maximum value `params.physical.maxA`.
- `prescale_bounds::Vector{Tuple{F,F}}`: Vector of tuples where each tuple defines
    the lower and upper bounds of the input for scaling.

# Returns
- `Y_law`: A `Law{Array{Float64, 2}}` instance that computes the hybrid diffusivity `Y`
    based on an input temperature and ice thickness using the neural network. The
    law scales the output to the physical bounds defined by `params`.

# Notes
- The computation is compatible with Zygote for automatic differentiation.

# Details
- The function wraps the architecture and state of the neural network in a`StatefulLuxLayer`.
- The resulting law takes input variables, applies the neural network, and scales its output
    to match the maximum value which is either `max_NN` or `params.physical.maxA`.
- The in-place assignment to `cache` is ignored in differentiation to allow gradient
    computation with Zygote.
- The `init_cache` function initializes the cache with a zero matrix.

# Example
```julia
nn_model = NeuralNetwork(params)
bounds_T = (-25.0, 0.0)
bounds_H = (0.0, 500.0)
Y_law = LawY(nn_model, params; prescale_bounds = [bounds_T, bounds_H])
"""
function LawY(
    nn_model::NeuralNetwork,
    params::Sleipnir.Parameters;
    max_NN::Union{F, Nothing} = nothing,
    prescale_bounds::Vector{Tuple{F,F}} = [(-25.0, 0.0), (0.0, 500.0)],
) where {F <: AbstractFloat}
    prescale = !isnothing(prescale_bounds) ? X -> _ml_model_prescale(X, prescale_bounds) : identity
    max_NN = isnothing(max_NN) ? params.physical.maxA : max_NN
    postscale = Y -> _ml_model_postscale(Y, max_NN)

    archi = nn_model.architecture
    st = nn_model.st
    smodel = StatefulLuxLayer{true}(archi, nothing, st)

    Y_law = let smodel = smodel, prescale = prescale, postscale = postscale
    Law{Array{Float64, 2}}(;
        inputs = (; T=iTemp(), H̄=iH̄()),
        f! = function (cache, inp, θ)
            A = map(h -> _pred_NN([inp.T, h], smodel, θ.Y, prescale, postscale), inp.H̄)

            # Flag the in-place assignment as non differented and return A instead in
            # order to be able to compute ∂A∂θ with Zygote
            Zygote.@ignore cache .= A
            return A
        end,
        init_cache = function (simulation, glacier_idx, θ; scalar::Bool = true)
            (; nx, ny) = simulation.glaciers[glacier_idx]
            return zeros(nx-1, ny-1)
        end,
    )
    end
    return Y_law
end

"""

    LawA(
        nn_model::NeuralNetwork,
        params::Sleipnir.Parameters,
    )

Constructs a law object for the creep coefficient `A` in the SIA based on a neural
network that takes as input the long term air temperature.
The creep coefficient `A` with this law is a scalar.
See also `SIA2D_A_target`.

# Arguments
- `nn_model::NeuralNetwork`: A neural network model containing the architecture
    `archi` and state `st` used for evaluation of the law.
- `params::Sleipnir.Parameters`: Parameters struct used to retrieve the minimum and
    maximum values of A for scaling of the neural network output.

# Returns
- `A_law`: A `Law{Array{Float64, 0}}` instance that computes the creep coefficient `A`
    based on an input temperature using the neural network. The law scales the
    output to the physical bounds defined by `params`.

# Notes
- The computation is compatible with Zygote for automatic differentiation.

# Details
- The function wraps the architecture and state of the neural network in a`StatefulLuxLayer`.
- The resulting law takes input variables, applies the neural network, and scales its output
    to be between `params.physical.minA` and `params.physical.maxA`.
- The in-place assignment to `cache` is ignored in differentiation to allow gradient
    computation with Zygote.
- The `init_cache` function initializes the cache with a scalar zero.

# Example
```julia
nn_model = NeuralNetwork(params)
A_law = LawA(nn_model, params)
"""
function LawA(
    nn_model::NeuralNetwork,
    params::Sleipnir.Parameters,
)
    archi = nn_model.architecture
    st = nn_model.st
    smodel = StatefulLuxLayer{true}(archi, nothing, st)

    A_law = let smodel = smodel, params = params
        Law{Array{Float64, 0}}(;
            inputs = (; T=iTemp()),
            f! = function (cache, inp, θ)
                min_NN = params.physical.minA
                max_NN = params.physical.maxA
                inp = collect(values(inp))
                A = only(scale(smodel(inp, θ.A), (min_NN, max_NN)))

                # Flag the in-place assignment as non differented and return A instead in
                # order to be able to compute ∂A∂θ with Zygote
                Zygote.@ignore cache .= A
                return A
            end,
            init_cache = function (simulation, glacier_idx, θ; scalar::Bool = false)
                return zeros()
            end,
        )
    end
    return A_law
end

include("laws_utils.jl")
include("laws_plots.jl")
