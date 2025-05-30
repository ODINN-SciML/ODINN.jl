### Utils used for the neural network architecture

export fourier_feature
export LuxFunction
export pretraining

"""
    fourier_feature(v, n::Integer=10, random=false, σ=5.0)

Generates a Fourier feature embedding of a vector `v`, optionally using randomized projections.

# Arguments
- `v`: Input vector to be transformed (typically a coordinate or feature vector).
- `n::Integer=10`: Number of Fourier features to generate (default is 10).
- `random::Bool=false`: Whether to use random Fourier features (default is `false`).
- `σ::Float64=5.0`: Standard deviation of the normal distribution used for random feature projection (only used if `random=true`).

# Returns
- A `2n`-dimensional vector consisting of sine and cosine features of the transformed input vector.

# Notes
Fourier features help to overcome spectral bias in neural networks and can further help to
learn higher frequncy components of the function faster. For more information, see
Tancik et. al (2020), "Fourier Features Let Networks Learn High Frequency Functions in Low
Dimensional Domains".

# Example
```julia
v = [0.5, 1.0]
features = fourier_feature(v, n=4, random=true, σ=2.0)

"""
function fourier_feature(v, n::Integer = 10, random = false, σ = 5.0)
    a₁ = ones(n)
    b₁ = ones(n)
    if random
        W = rand(Normal(0, σ), (n, length(v)))
    else
        W = 1.0:1.0:n |> collect
    end

    return [a₁ .* sin.(π .* W .* v); b₁ .* cos.(π .* W .* v)]
end

"""
This function allows to extend the Wrapper layers define in Lux to matrices operations.
"""
function LuxFunction(f::Function, v::Union{Vector,SubArray})
    return f(v)
end

function LuxFunction(f::Function, V::Matrix)
    return reduce(hcat, map(v -> f(v), eachcol(V)))
end

"""
    pretraining(architecture::Lux.Chain;
                X::Matrix,
                Y::Matrix,
                nepochs::Int=3000,
                lossfn::GenericLossFunction=MSLELoss(; agg=mean, epsilon=1e-10),
                rng::AbstractRNG=Random.default_rng())

Pretrains a neural network model using a input and output.

# Arguments
- `architecture::Lux.Chain`: The neural network architecture to be trained.
- `X::Matrix`: Input feature matrix where each column is a feature vector.
- `Y::Matrix`: Target output matrix corresponding to the inputs in `X`.
- `nepochs::Int=3000`: Number of training epochs (default is 3000).
- `lossfn::GenericLossFunction=MSLELoss(...)`: Loss function used for training. Defaults to Mean Squared Logarithmic Error.
- `rng::AbstractRNG=Random.default_rng()`: Random number generator used for parameter initialization.

# Returns
- `architecture`: The trained neural network architecture.
- `θ_pretrain`: Trained parameters of the neural network.
- `st_pretrain`: Internal states of the trained model.

# Notes
Pretrainign helps to reduce the number of total epochs required to train the UDE by
selecting a physical meaningful initialization for the model.
The function initializes the model parameters and states using `Lux.setup`, then performs
training using a custom `train_model!` function with ADAM optimizer. Loss values are printed
every 100 epochs during training.

# Example
```julia
using Lux, Random
arch = Chain(Dense(10 => 20, relu), Dense(20 => 1))
X = rand(10, 100)
Y = rand(1, 100)
model, params, state = pretraining(arch; X=X, Y=Y)
"""
function pretraining(
    architecture::Lux.Chain;
    X::Matrix,
    Y::Matrix,
    nepochs::Int = 3000,
    lossfn::GenericLossFunction = MSLELoss(; agg = mean, epsilon = 1e-10),
    rng::AbstractRNG = Random.default_rng(),
    )

    @info "Pretrainign neural network with initial guess for diffusivity."

    function train_model!(model, ps, st, opt, nepochs::Int)
        tstate = Training.TrainState(model, ps, st, opt)
        for i in 1:nepochs
            grads, loss, _, tstate = Training.single_train_step!(
                AutoZygote(), lossfn, (X, Y), tstate
            )
            if i % 100 == 0 || i == 1 || i == nepochs
                ODINN.@printf "Loss Value after %6d iterations: %.8f\n" i loss
            end
        end
        return tstate.model, tstate.parameters, tstate.states
    end

    θ_setup, st_setup = Lux.setup(rng, architecture)
    θ_setup = ComponentArray(θ_setup)
    _architecture, θ_pretrain, st_pretrain = train_model!(
        architecture, θ_setup, st_setup,
        ODINN.ADAM(), nepochs
        )

    return _architecture, θ_pretrain, st_pretrain
end