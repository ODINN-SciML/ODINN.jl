const t₁ = 5                      # number of simulation years 
const ρ = 900                     # Ice density [kg / m^3]
const g = 9.81                    # Gravitational acceleration [m / s^2]
const n = 3                       # Glen's flow law exponent
const ϵ = 1e-4                    # small number
const maxA = 8e-16
const minA = 3e-17
const maxT = 1
const minT = -25

create_ref_dataset = false          # Run reference PDE to generate reference dataset
loss_type = "H"                   # Loss function based on ice thickness
# loss_type = "V"                   # Loss function based on ice surface velocities
# loss_type = "HV"                    # Combined loss function based on ice surface velocities and ice thickness
const random_sampling_loss = false  # Use random subset of matrix samples for the loss
const norm_loss = false
const noise = true                  # Add random noise to fake A law
rng_seed() = MersenneTwister(123)   # Random seed

const minA_out = 0.3
const maxA_out = 8
