
# Abstract type as a parent type for simulations
abstract type Simulation end

include("results/Results.jl")
include("predictions/Prediction.jl")
include("inversions/Inversion.jl")
include("functional_inversions/FunctionalInversion.jl")

###############################################
################### UTILS #####################
###############################################

include(joinpath(ODINN.root_dir, "src/models/iceflow/SIA2D/SIA2D_utils.jl"))
include(joinpath(ODINN.root_dir, "src/simulations/simulation_utils.jl"))
include(joinpath(ODINN.root_dir, "src/simulations/results/results_utils.jl"))


