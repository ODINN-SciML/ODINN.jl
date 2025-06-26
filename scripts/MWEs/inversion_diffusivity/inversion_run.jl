using Pkg; Pkg.activate(".")
include("inversion_setup.jl")

# We run the simulation with ADAM and then LBFGS
run!(
    functional_inversion;
    path = joinpath(ODINN.root_dir, "scripts/MWEs/inversion_diffusivity/data"),
    file_name = "simulation_result_Halfar.jld2"
    )