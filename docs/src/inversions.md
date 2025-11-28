# Inversion types

In this section we present the main ideas behind the inversions.
In data assimilation, one aims at calibrating a model such that the predictions match the observations.
The observations can be either an initial, intermediate or final state or observations over time. When we try to invert a trajectory by matching multiple states or observations over time, we are referring to a transient inversion. 

In the classical setting, inversions are made with respect to some quantity of interest that is involed in the iceflow equation, i.e. the mechanistic model.
This can be for example the creep coefficient $A$ for the Shallow Ice Approximation (SIA) like in [bolibar_universal_2023](@cite).
This parameter can be constant for one glacier or vary with respect to space (i.e. in a grid), in which case a spatial regularization is added to make the inversion problem well-posed.

Another way to invert a parameter in an iceflow equation is to parametrize the quantity of interest, let us say $A$, by other quantities which can be in the case of [bolibar_universal_2023](@cite), the long term air temperature $T$.
The optimized variable is not $A$ but the parameters $\theta$ and the mapping $(T,\theta)\mapsto A(\theta,T)$ defines a parametrization of the ice rheology. We refer hereafter to this kind of inversion as a *functional inversion*, where the goal is to optimize the parameters of a regressor specifying a function, rather than directly optimizing parameters or coefficients of a mechanistic model.

We summarize the main differences between the two kind of inversions hereafter, and we specify the case for performing inversions of the initial conditions (i.e. glacier state), which are a sub-case of classical inversions.

## Classical inversions

We refer to classical inversions as the inverse problems where the objective is to directly invert the parameters $p$ of a mechanistic model itself (e.g. the SIA equation). These type of inversions are handled in ODINN via the `Inversion` subtype of `Simulation`.

The optimization problem is
```math
\min_\theta L \left(\hat S, S \right) + R\left( \hat S, p \right)
```
where $\theta$ is the vector of parameters to invert, for example $\theta=[A]$; $L(\hat S, S)$ is the empirical loss function between a predicted state $\hat S$ and observations $S$ (for example, the ice thickness or the ice surface velocity); and $R(\hat S, p)$ is a regularization term.

### Classical inversions tutorial

The [classical inversion tutorial](./classical_inversion.md) provides an example on how to use these methods to invert in a scalar or gridded way a parameter of the Shallow Ice Approximation. 

### Inversion with respect to initial state of the glacier

A specific case of a classical inversion is inverting the initial state of glacier. Ice flow models are very sensitive to the initial state of the glacier. For example, it is very common to observe how bad initializations of the ice thickness can lead to unrealistic physical evolutions of the glacier over time (see [Perego_Price_Stadler_2014](@cite) for a good explanation of this phenomenon in the context of ice sheet simulations). For cases where the initial condition of the glacier is not completely known and/or we want to avoid numerical shocks during the forward simulation, ODINN provides an interface to optimize the initial conditions of the ice thickness $H(t = t_0) = H_0$ as an extra parameter of the inversion (e.g. in conjunction with the coefficients of basal sliding or the parameters of a neural network parametrization).

We provide an object `InitialCondition` that enables the prescription of all the important parameters of the initialization. An initial guess can be provided: `:Farinotti2019` for the ice thickness product by Farinotti et al. (2019), or `:Farinotti2019Random` for the same product with some added Gaussian noise. In order to guarantee that the ice thickness is always non-negative, we introduce a filter function that maps model parameters into a non-negative initial ice thickness matrix (see `evaluate_H₀`).
Finally, prescribing an inversion with respect to the initial condition can be directly included by defining the initial condition as one extra regressor inside the model:

```julia
ic = InitialCondition(params, glaciers, :Farinotti2019)
model = Model(
    iceflow = SIA2Dmodel(params; A = LawA(nn_model, params)),
    mass_balance = nothing,
    regressors = (; A = nn_model, IC = ic)
)
```

From an optimization perspective, the computation of the gradients of the loss function with respect to the initial condition is simply given by the value of the adjoint variable at the initial time, which is part of the standard inversion pipeline of ODINN and it does not add any extra computational overhead.

Once the model has been trained, the value of the initialization can be computed directly from the optimized parameter `θ` using the `evaluate_H₀` function on a specific glacier.

## Functional inversions

We refer to functional inversions as the inverse problems where the objective is to invert the parameter $\theta$ of a regressor (e.g. a neural network), in order to learn a function that parametrizes a subpart of a mechanistic model (e.g. the SIA) with respect to one or more input variables (e.g. surface melt, basal slope) [bolibar_universal_2023](@cite). The methods behind functional inversions are known as **Universal Differential Equations** [rackauckas_universal_2020](@cite).

We present the concept of a functional inversion for the case where we want to learn a law for the ice rheology $A$ in the Shallow Ice Approximation by using a neural network as a parametrization $A=\text{NN}(\theta,T)$ with weights $\theta$.
```math
\begin{aligned}& \min_\theta \mathcal{D}\left(\hat S, S \right) + \mathcal{R}\left( \hat S, p \right)\\
& A=\text{NN}(\theta,T)
\end{aligned}
```

![Overview of ODINN.jl’s workflow to perform functional inversions of glacier physical processes using Universal Differential Equations.](assets/overview_figure.png)

> **Figure:** Overview of `ODINN.jl`’s workflow to perform functional inversions of glacier physical processes using Universal Differential Equations. The parameters ($θ$) of a function determining a given physical process ($D_θ$), expressed by a neural network $NN_θ$, are optimized in order to minimize a loss function. In this example, the physical to be inferred law was constrained only by climate data, but any other proxies of interest can be used to design it. The climate data, and therefore the glacier mass balance, are downscaled (i.e. it depends on $S$), with $S$ being updated by the solver, thus dynamically updating the state of the simulation for a given timestep.

### Understanding the `Law`s interface

In ODINN, the components of the iceflow equation can be customized with a `Law` type.
It is responsible for linking a given regressor and a set of input variables to a target component of a mechanistic model (for now the SIA).
Here is a quick example on how this looks like:

```julia
law_inputs = (; CPDD = iCPDD(), topo_roughness = iTopoRough())
model = Model(
    iceflow = SIA2Dmodel(params; C=SyntheticC(params; inputs=law_inputs)),
    mass_balance = nothing
)
```

In this piece of code, we are selecting cumulative positive degree days (CPDDs) and topographical roughness as inputs for a law/parametrization named `SyntheticC`. Then, when declaring the ice flow model, we associate it to the parameter `C` of the iceflow model (i.e. the basal sliding). Using this simple interface, we can easily combine all sorts of input variables, with different laws and targets/subparts of mechanistic models.

!!! warning
    It is important to bear in mind that new input types and laws cannot be created on the fly, they need to be specified/added by a user beforehand. For input variables, it is generally a matter of fetching the right data and processing it to the right format for the law/function. For laws, one just needs to specify which function is applied to the different input variables. If the law and input variables involve a regressor (that is a learnable component), then the new types for the law and inputs must be added to `ODINN.jl` [here](https://github.com/ODINN-SciML/ODINN.jl/blob/main/src/laws/Laws.jl); if the law doesn't include any regressors, they can be added directly to `Huginn.jl` [here](https://github.com/ODINN-SciML/Huginn.jl/blob/main/src/laws/Laws.jl).

Here is an example of how the code of an input variable looks like:

```julia
# We first need to declare the type for the input variable, with any fields that might be needed
struct iCPDD{P<:Period} <: AbstractInput
    window::P
    function iCPDD(; window::P = Week(1)) where {P<:Period}
        new{typeof(window)}(window)
    end
end
default_name(::iCPDD) = :CPDD

# And then, using multiple dispatch, we specify the righ `get_input` function for this type, i.e. how to get it
function get_input(cpdd::iCPDD, simulation, glacier_idx, t)  
    window = cpdd.window  
    glacier = simulation.glaciers[glacier_idx]  
    # We trim only the time period between `t` and `t - x`, where `x` is the PDD time window defined in the input attributes. 
    period = (partial_year(Day, t) - window):Day(1):partial_year(Day, t)  
    get_cumulative_climate!(glacier.climate, period)  
    # Convert climate dataset to 2D based on the glacier's DEM  
    climate_2D_step = downscale_2D_climate(glacier.climate.climate_step, glacier.S, glacier.Coords)  

    return climate_2D_step.PDD
end
function Base.zero(::iCPDD, simulation, glacier_idx)
    (; nx, ny) = simulation.glaciers[glacier_idx]
    return zeros(nx, ny)
end
```

A synthetic `C` law can then be defined using this `iCPDD` input.
Here is a simple example of a synthetic law made following this interface:

```julia
function SyntheticC(params::Sleipnir.Parameters; inputs = (; CPDD=iCPDD()))
    C_synth_law = Law{Array{Float64, 2}}(;
        name = :SyntheticC,
        inputs = inputs,
        max_value = params.physical.maxC,
        min_value = params.physical.minC,
        f! = function (cache, inp, θ)
            # Nonlinear scaling using a sigmoid transformation
            # C = Cmin + (Cmax - Cmin) * sigmoid(β * (inp.CPDD)))
            # β controls the steepness of the sigmoid, ϵ avoids division by zero
            Cmin = params.physical.minC
            Cmax = params.physical.maxC
            β = 1.0      # Steepness parameter for sigmoid
            sigmoid = @. 1.0 / (1.0 + exp(-β * (inp.CPDD - 1.0)))  # Center sigmoid at x=1 for flexibility
            # If the provided C values are a matrix, reduce matrix size to match operations
            cache .= Cmin .+ (Cmax - Cmin) .* (isa(sigmoid, Matrix) ? inn1(sigmoid) : sigmoid)
        end,
        init_cache = function (simulation, glacier_idx, θ)
            zeros(size(simulation.glaciers[glacier_idx].S) .- 1)
        end,
        callback_freq = 1/52,  # weekly frequency
    )
    return C_synth_law
end
```

Creating a `Law` implies declaring different components which are provided to the constructor function:
- (1) `inputs`: The inputs that the law uses, which is provided as a named tuple. The values of the named tuple are subtypes of `AbstractInput`.
- (2) `f!`: The function that applies the law.
- (3) `init_cache`: A function that describes how the cache needs to be initialized for the law to interact with the simulation. It returns an initialized cache.
- (4) `callback_freq`: Optionally the callback frequency which determines the time frequency on which the law will be called during the simulation (e.g. weekly).
It is also possible to not provide the `inputs` in which case the `f!` function is in charge of retrieving the appropriate variables to use in the law.
See in the API the docstring of [`Sleipnir.Law`](./api.md#Sleipnir.Law-api).

Functional inversions in ODINN are also handled by the `Inversion` subtype of `Simulation`.
Whether a classical or a functional inversion should be made with respect to some of the components of the PDE depends on how the law is defined.

### Functional inversion tutorial

The [functional inversion tutorial](./functional_inversion.md) gives an example of how such an inversion can be run in practice with `ODINN.jl`.

### Laws and Law inputs tutorials

For more details on how to actually implement new laws and law inputs, and for a list of all the ones that are already available within `ODINN.jl`, you can check the [Laws tutorial](./laws.md) and the [Laws inputs tutorial](./input_laws.md) respectively. 
