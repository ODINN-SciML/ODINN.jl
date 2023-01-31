export t₁, current_epoch, loss_history

# const t₁ = 5.0                  # number of simulation years 
const ρ = Ref{Float64}(900.0)                 # Ice density [kg / m^3]
const g = Ref{Float64}(9.81)                  # Gravitational acceleration [m / s^2]
const n = Ref{Float64}(3.0)                   # Glen's flow law exponent
const ϵ = Ref{Float64}(1e-3)                    # small number
const maxA = Ref{Float64}(8e-17)
const minA = Ref{Float64}(8.5e-20)
const maxT = Ref{Float64}(1.0)
const minT = Ref{Float64}(-25.0)

const base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands/"
# const base_url = "https://cluster.klima.uni-bremen.de/~fmaussion/share/jordi/prepro_dir_v0/"

# From Cuffey and Paterson
const A_values_sec = ([0.0 -2.0 -5.0 -10.0 -15.0 -20.0 -25.0 -30.0 -35.0 -40.0 -45.0 -50.0;
                              2.4e-24 1.7e-24 9.3e-25 3.5e-25 2.1e-25 1.2e-25 6.8e-26 3.7e-26 2.0e-26 1.0e-26 5.2e-27 2.6e-27]) # s⁻¹Pa⁻³
const A_values = hcat(A_values_sec[1,:], A_values_sec[2,:].*60.0*60.0*24.0*365.25)'

const noise_A_magnitude = Ref{Float64}(5e-18)  # magnitude of noise to be added to A

# Mass balance references for max and min values, used for random MB generation
const ref_max_MB = Ref{Float64}(5.0)
const ref_min_MB = Ref{Float64}(-10.0)
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
current_minibatches = 0
loss_epoch = 0.0
loss_history = []
global optimization_method = "AD+AD"