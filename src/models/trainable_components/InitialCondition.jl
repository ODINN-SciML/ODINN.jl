export InitialCondition

# Empty initial condition
struct emptyIC <: PerGlacierModel end

"""
    InitialCondition{
        ComponentVectorType <: ComponentVector
    } <: PerGlacierModel

Per glacier initial condition container.
`InitialCondition` wraps a ComponentVector (θ) that stores one matrix per glacier and implements the `InitialCondition` interface used by the inversion machinery.

# Fields

  - `θ::ComponentVectorType`: The per glacier matrix of initial condition.

# Constructor

    InitialCondition(
        params::Sleipnir.Parameters,
        glaciers::Vector{<: AbstractGlacier},
        initialization::Symbol = :Farinotti2019,
    )

# Arguments

  - `params::Sleipnir.Parameters`: Parameters struct.
  - `glaciers::Vector{<: AbstractGlacier}`: Vector of AbstractGlacier. The i-th entry in θ corresponds to glaciers[i].
  - `initialization::Symbol`: Symbol providing the way the initial condition should be initialized.

# Example

```julia
InitialCondition(params, glaciers, :Farinotti2019)
```
"""
mutable struct InitialCondition{
    ComponentVectorType <: ComponentVector
} <: PerGlacierModel
    θ::ComponentVectorType

    function InitialCondition(
            params::Sleipnir.Parameters,
            glaciers::Vector{<: AbstractGlacier},
            initialization::Symbol = :Farinotti2019
    )
        # Float type
        ft = Sleipnir.Float
        # Component Array type
        initial_condition_type = Tuple(Symbol("$(i)") for i in 1:length(glaciers))

        # Define a series of initial conditions
        if initialization == :Farinotti2019
            initial_condition = NamedTuple{initial_condition_type}(
                Tuple(glaciers[i].H₀ for i in 1:length(glaciers))
            )
        elseif initialization == :Farinotti2019Random
            stdH = 10.0
            grid_length = 10
            initial_condition = NamedTuple{initial_condition_type}(
                Tuple(random_matrix(glaciers[i].H₀, stdH, grid_length)
            for i in 1:length(glaciers))
            )
        else
            @error "Strategy for initialization of ice thicknesses not found."
        end

        θ = ComponentVector{ft}(θ = initial_condition)
        new{typeof(θ)}(θ)
    end
end

# Display setup
function Base.show(io::IO, invertible_model::InitialCondition)
    println(io, "--- Initial condition to invert ---")
    println(io, "    Matrix per glacier")
    print(io, "    θ: ComponentVector of length $(length(invertible_model.θ))")
end

include("./InitialCondition_utils.jl")
