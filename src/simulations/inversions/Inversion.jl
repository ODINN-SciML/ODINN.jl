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
mutable struct Inversion{
    MODEL <: Sleipnir.Model,
    CACHE,
    GLACIER <: Sleipnir.AbstractGlacier,
    PARAMS <: Sleipnir.Parameters,
    RES <: ODINN.Results
} <: Simulation
    model::MODEL
    cache::Union{CACHE, Nothing}
    glaciers::Vector{GLACIER}
    parameters::PARAMS
    results::RES
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

    # Optionally calibrate the mass balance model per glacier (no-op unless the
    # model type defines a calibration routine, e.g. TImodel1).
    if parameters.simulation.use_MB && parameters.simulation.calibrate_MB
        calibrate_MB_model!(model, glaciers, parameters)
    end

    # Build the results struct based on input values
    emptySimulationResults = Vector{Sleipnir.Results{Sleipnir.Float, Sleipnir.Int}}([])
    emptyResults = Results(emptySimulationResults, TrainingStats())
    inversion = Inversion{M, cache_type(model), G, typeof(parameters), typeof(emptyResults)}(
        model, nothing,
        glaciers,
        parameters,
        emptyResults)

    return inversion
end

# Display setup
Base.show(io::IO, ::MIME"text/plain", inversion::Inversion) = Base.show(io, inversion)
function Base.show(io::IO, inversion::Inversion)
    pad = 14

    println(io, "Inversion")

    # ── glaciers ──────────────────────────────────────────────────────────────
    label(io, "  glaciers", pad)
    n = length(inversion.glaciers)
    val(io, "$n");
    hint(io, " $(n == 1 ? "glacier" : "glaciers")")
    println(io)

    # ── model ─────────────────────────────────────────────────────────────────
    label(io, "  model", pad)
    field(io, "iceflow");
    print(io, " = ")
    val(io, "$(nameof(typeof(inversion.model.iceflow)))")
    sep(io)
    field(io, "mass_balance");
    print(io, " = ")
    val(io, "$(nameof(typeof(inversion.model.mass_balance)))")
    sep(io)
    field(io, "learnable");
    print(io, " =")
    if isnothing(inversion.model.trainable_components)
        hint(io, " (nothing)")
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
    label(io, "  parameters", pad)
    println(io)
    params_str = sprint(show, inversion.parameters)
    for line in split(params_str, "\n")
        isempty(line) && continue
        occursin(r"^Parameters$", line) && continue
        printstyled(io, "    "; color = :light_black)
        println(io, line)
    end

    # ── cache ─────────────────────────────────────────────────────────────────
    label(io, "  cache", pad)
    if isnothing(inversion.cache)
        hint(io, "(nothing)")
    else
        val(io, "$(nameof(typeof(inversion.cache)))")
    end
    println(io)

    # ── results ───────────────────────────────────────────────────────────────
    label(io, "  results", pad)
    stats = inversion.results.stats
    if stats.niter == 0
        print(io, check(false));
        hint(io, " not yet run")
    else
        total_epochs = inversion.parameters.hyper.epochs isa Vector ?
                       sum(inversion.parameters.hyper.epochs) :
                       inversion.parameters.hyper.epochs
        print(io, check(true))
        field(io, " epoch");
        print(io, " = ");
        val(io, "$(stats.niter)")
        hint(io, " / $total_epochs")
        sep(io)
        field(io, "loss");
        print(io, " = ");
        val(io, "$(last(stats.losses))")
        sep(io)
        field(io, "#(simulation)");
        print(io, " = ")
        val(io, "$(length(inversion.results.simulation))")
    end
    println(io)
end

###############################################
################### UTILS #####################
###############################################

include("sciml_utils.jl")
include("inversion_utils.jl")
include("callback_utils.jl")
