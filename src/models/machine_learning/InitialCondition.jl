export InitialCondition

include("./InitialCondition_utils.jl")

# Empty initial condition
struct emptyIC <: MLmodel end
"""

"""
mutable struct InitialCondition{
    ComponentVectorType <: ComponentVector
    } <: MLmodel
    θ::ComponentVectorType

    function InitialCondition(
        params::P,
        glaciers::Vector{G},
        initialization::Symbol = :Farinotti2019,
    ) where {
        P<:Sleipnir.Parameters,
        G<:AbstractGlacier
    }
        # Float type
        ft = Sleipnir.Float
        # Component Array type
        initial_condition_type = Tuple(Symbol("$(glaciers[i].rgi_id)") for i in 1:length(glaciers))

        # Define a series of initial conditions
        if initialization == :Farinotti2019
            initial_condition = NamedTuple{initial_condition_type}(
                Tuple(glaciers[i].H₀ for i in 1:length(glaciers))
                )
        elseif initialization == :Farinotti2019Random
            stdH = 10.0
            grid_length = 10
            initial_condition = NamedTuple{initial_condition_type}(
                Tuple(random_matrix(glaciers[i].H₀, stdH, grid_length) for i in 1:length(glaciers))
                )
        else
            @error "Strategy for initialization of ice thicknesses not found."
        end

        θ = ComponentVector{ft}(θ = initial_condition)
        new{typeof(θ)}(
            θ
        )
    end

end