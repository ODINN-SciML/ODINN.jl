export t₁, current_epoch, loss_history

# const t₁ = 5.0                  # number of simulation years 
const ρ = Ref{Float32}(900.0f0)                 # Ice density [kg / m^3]
const g = Ref{Float32}(9.81f0)                  # Gravitational acceleration [m / s^2]
const n = Ref{Float32}(3.0f0)                   # Glen's flow law exponent
const ϵ = Ref{Float32}(1f-3)                    # small number
const maxA = Ref{Float32}(8f-17)
const minA = Ref{Float32}(8.5f-20)
const maxT = Ref{Float32}(1.0f0)
const minT = Ref{Float32}(-25.0f0)

const base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands/"
# const base_url = "https://cluster.klima.uni-bremen.de/~fmaussion/share/jordi/prepro_dir_v0/"

# From Cuffey and Paterson
const A_values_sec = ([0.0f0 -2.0f0 -5.0f0 -10.0f0 -15.0f0 -20.0f0 -25.0f0 -30.0f0 -35.0f0 -40.0f0 -45.0f0 -50.0f0;
                              2.4f-24 1.7f-24 9.3f-25 3.5f-25 2.1f-25 1.2f-25 6.8f-26 3.7f-26 2.0f-26 1.0f-26 5.2f-27 2.6f-27]) # s⁻¹Pa⁻³
const A_values = hcat(A_values_sec[1,:], A_values_sec[2,:].*60.0f0*60.0f0*24.0f0*365.25f0)'

const noise_A_magnitude = Ref{Float32}(5f-18)  # magnitude of noise to be added to A

# Mass balance references for max and min values, used for random MB generation
const ref_max_MB = Ref{Float32}(5.0f0)
const ref_min_MB = Ref{Float32}(-10.0f0)
use_MB = true
# Spin up and reference simulations
run_spinup = false
use_spinup = false
create_ref_dataset = true
# UDE training
train = true
retrain = false

plots = true                     # Make reference plots 
const overwrite_climate = false         # Force re-computing climate data for glaciers
# loss_type = "H"                   # Loss function based on ice thickness
const loss_type = Ref{String}("V")                   # Loss function based on ice surface velocities
# loss_type = "HV"                    # Combined loss function based on ice surface velocities and ice thickness
const random_sampling_loss = Ref{Bool}(false)  # Use random subset of matrix samples for the loss
const scale_loss = Ref{Bool}(true)
const noise = Ref{Bool}(true)                  # Add random noise to fake A law
rng_seed() = MersenneTwister(1010)   # Random seed

ice_thickness_source = "farinotti" # choose between "farinotti" or "millan"

# Machine learning training
current_epoch = 1
loss_history = []
global optimization_method = "AD+AD"