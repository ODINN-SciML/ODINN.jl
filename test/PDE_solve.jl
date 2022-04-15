

oggm_config()

## Retrieving gdirs and climate for the following glaciers  
rgi_ids = ["RGI60-11.03638", "RGI60-11.01450", "RGI60-08.00213", "RGI60-04.04351", "RGI60-01.02170",
"RGI60-02.05098", "RGI60-01.01104", "RGI60-01.09162", "RGI60-01.00570", "RGI60-04.07051",                	
"RGI60-07.00274", "RGI60-07.01323", "RGI60-03.04207", "RGI60-03.03533", "RGI60-01.17316"]

gdirs = init_gdirs(rgi_ids, force=false)
tspan = (0.0,5.0) # period in years for simulation
gdirs_climate = get_gdirs_with_climate(gdirs, tspan, overwrite=false, plot=false)

# Load input data for the SIA PDE simulation
# gdirs_climate = JLD.load(joinpath(ODINN.root_dir, "test/data/gdirs_climate.jld"), "gdirs_climate")
# Load reference values for the simulation
PDE_refs = load(joinpath(ODINN.root_dir, "data/PDE_refs.jld2"))

# Run the forward PDE simulation
H_refs, V̄x_refs, V̄y_refs = @time generate_ref_dataset(gdirs_climate, tspan)

@test H_refs ≈ PDE_refs["H_refs"]
@test V̄x_refs ≈ PDE_refs["V̄x_refs"]
@test V̄y_refs ≈ PDE_refs["V̄y_refs"]