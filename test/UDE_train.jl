
# Load reference values for the simulation
PDE_refs = load(joinpath(ODINN.root_dir, "data/PDE_refs.jld2"))
train_settings = (ADAM(0.03), 10) # optimizer, epochs
iceflow_trained,
UA = @time train_iceflow_UDE(gdirs_climate, tspan, train_settings, PDE_refs)

# Verify neural PDE is learning
@test loss_history[1] > 3*loss_history[end]
