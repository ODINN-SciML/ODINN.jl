export Inversion

"""
    mutable struct Inversion{MODEL, CACHE, GLACIER, RES} <: Simulation

An object representing an inversion simulation.
It can involve at the same time a classical inversion and a functional inversion (i.e. the inversion of a function using some data-driven regressor).

# Fields

  - `model::Sleipnir.Model`: The model used for the simulation.
  - `glaciers::Vector{Sleipnir.AbstractGlacier}`: A vector of glaciers involved in the simulation.
  - `parameters::Sleipnir.Parameters`: The parameters used for the simulation.
  - `results::ODINN.Results`: A `ODINN.Results` instance to store the results of the inversion and of the forward simulations.
"""
mutable struct Inversion{MODEL, CACHE, GLACIER, RES} <: Simulation
    model::MODEL
    cache::Union{CACHE, Nothing}
    glaciers::Vector{GLACIER}
    parameters::Sleipnir.Parameters
    results::ODINN.Results
end

"""
    function Inversion(
        model::M,
        glaciers::Vector{G},
        parameters::P
    ) where {G <: Sleipnir.AbstractGlacier, M <: Sleipnir.Model, P <: Sleipnir.Parameters}

Constructor for Inversion struct with glacier model information, glaciers, and parameters.

# Arguments

  - `model::Sleipnir.Model`: The model used for the simulation.
  - `glaciers::Vector{G}`: A vector of glaciers involved in the simulation.
  - `parameters::Sleipnir.Parameters`: The parameters used for the simulation.

# Returns

  - `Inversion`: A new instance of the Inversion struct.
"""
function Inversion(
        model::M,
        glaciers::Vector{G},
        parameters::P
) where {G <: Sleipnir.AbstractGlacier, M <: Sleipnir.Model, P <: Sleipnir.Parameters}

    # We perform this check here to avoid having to provide the parameters when creating the model
    @assert targetType(model.trainable_components.target) == parameters.UDE.target "Target does not match the one provided in the parameters."
    Muninn.validate_model_simulation_compatibility(model, parameters)

    # Build the results struct based on input values
    emptySimulationResults = Vector{Sleipnir.Results{Sleipnir.Float, Sleipnir.Int}}([])
    emptyResults = Results(emptySimulationResults, TrainingStats())
    inversion = Inversion{M, cache_type(model), G, typeof(emptyResults)}(model, nothing,
        glaciers,
        parameters,
        emptyResults)

    return inversion
end

# Display setup
Base.show(io::IO, ::MIME"text/plain", inversion::Inversion) = Base.show(io, inversion)
function Base.show(io::IO, inversion::Inversion)
    label(s) = printstyled(io, rpad(s, 14); color = :light_black)
    sep() = printstyled(io, " · "; color = :light_black)
    field(s) = printstyled(io, s; color = :light_black)
    val(s) = print(io, s)
    hint(s) = printstyled(io, s; color = :light_black)
    check(b) = b ? "\e[32m✓\e[0m" : "\e[31m✗\e[0m"

    println(io, "Inversion")

    # ── glaciers ──────────────────────────────────────────────────────────────
    label("  glaciers")
    n = length(inversion.glaciers)
    val("$n");
    hint(" $(n == 1 ? "glacier" : "glaciers")")
    println(io)

    # ── model ─────────────────────────────────────────────────────────────────
    label("  model")
    field("iceflow");
    print(io, " = ")
    val("$(nameof(typeof(inversion.model.iceflow)))")
    sep()
    field("mass_balance");
    print(io, " = ")
    val("$(nameof(typeof(inversion.model.mass_balance)))")
    sep()
    field("learnable");
    print(io, " =")
    if isnothing(inversion.model.trainable_components)
        hint(" (nothing)")
        println(io)
    else
        println(io)
        tc_str = sprint(show, inversion.model.trainable_components)
        for line in split(tc_str, "\n")
            isempty(line) && continue
            printstyled(io, "    "; color = :light_black)
            println(io, line)
        end
    end

    # ── parameters ────────────────────────────────────────────────────────────
    label("  parameters")
    println(io)
    params_str = sprint(show, inversion.parameters)
    for line in split(params_str, "\n")
        isempty(line) && continue
        occursin(r"^Parameters$", line) && continue
        printstyled(io, "    "; color = :light_black)
        println(io, line)
    end

    # ── cache ─────────────────────────────────────────────────────────────────
    label("  cache")
    if isnothing(inversion.cache)
        hint("(nothing)")
    else
        val("$(nameof(typeof(inversion.cache)))")
    end
    println(io)

    # ── results ───────────────────────────────────────────────────────────────
    label("  results")
    stats = inversion.results.stats
    if stats.niter == 0
        print(io, check(false));
        hint(" not yet run")
    else
        total_epochs = inversion.parameters.hyper.epochs isa Vector ?
                       sum(inversion.parameters.hyper.epochs) :
                       inversion.parameters.hyper.epochs
        print(io, check(true))
        field(" epoch");
        print(io, " = ");
        val("$(stats.niter)")
        hint(" / $total_epochs")
        sep()
        field("loss");
        print(io, " = ");
        val("$(last(stats.losses))")
        sep()
        field("#(simulation)");
        print(io, " = ")
        val("$(length(inversion.results.simulation))")
    end
    println(io)
end

###############################################
################### UTILS #####################
###############################################

include("sciml_utils.jl")
include("inversion_utils.jl")
include("callback_utils.jl")
