using Pkg
odinn_folder = split(Base.source_dir(), "scripts")[1]
Pkg.activate(odinn_folder*"/scripts/")

include("inversion_setup.jl")

# We run the simulation with ADAM and then LBFGS
run!(
    functional_inversion;
    path = joinpath(ODINN.root_dir, "scripts/MWEs/inversion_A/data"),
    file_name = "simulation_result.jld2"
)
