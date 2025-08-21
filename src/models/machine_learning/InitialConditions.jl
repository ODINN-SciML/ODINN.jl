export InitialCondition

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
        glaciers::Union{Vector{G}, Nothing} = nothing,
        initialization::Symbol = :Farinotti2019,
    ) where {
        P<:Sleipnir.Parameters,
        G<:AbstractGlacier
    }
        # Float type
        ft = Sleipnir.Float

        if params.inversion.train_initial_conditions
            # TODO: we do it for a single glacier now
            if initialization == :Farinotti2019
                θ = ComponentVector{ft}(θ = glaciers[1].H₀)
                # θ = ComponentVector{ft}(θ = [glacier.H₀ for glacier in glaciers])
            elseif initialization == :Farinotti2019Random
                H₀ = glaciers[1].H₀
                H₀ .*= max.(0.1 .* randn(size(H₀)...) .+ 1.0, 0.0)
                # H₀ .*= 0.90
                θ = ComponentVector{ft}(θ = H₀)
            else
                @error "Strategy for initialization of ice thicknesses not found."
            end
        else
            @error "We need to standardize how IC is specified"
            θ = nothing
        end

        new{typeof(θ)}(
            θ
        )
    end

end