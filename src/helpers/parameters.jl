export t₁, current_epoch, loss_history

# const t₁ = 5.0                      # number of simulation years 
const ρ = 900.0                     # Ice density [kg / m^3]
const g = 9.81                    # Gravitational acceleration [m / s^2]
const n = 3.0                       # Glen's flow law exponent
const ϵ = 1e-3                    # small number
const maxA = 8e-17
const minA = 8.5e-20
const maxT = 1.0
const minT = -25.0

# From Cuffey and Paterson
const A_values_sec = [0.0 -2.0 -5.0 -10.0 -15.0 -20.0 -25.0 -30.0 -35.0 -40.0 -45.0 -50.0;
                              2.4e-24 1.7e-24 9.3e-25 3.5e-25 2.1e-25 1.2e-25 6.8e-26 3.7e-26 2.0e-26 1.0e-26 5.2e-27 2.6e-27] # s⁻¹Pa⁻³
const A_values = hcat(A_values_sec[1,:], A_values_sec[2,:].*60*60*24*365.25)'

plots = false                      # Display plots 
overwrite_climate = false          # Force re-computing climate data for glaciers
# loss_type = "H"                   # Loss function based on ice thickness
loss_type = "V"                   # Loss function based on ice surface velocities
# loss_type = "HV"                    # Combined loss function based on ice surface velocities and ice thickness
const random_sampling_loss = false  # Use random subset of matrix samples for the loss
const scale_loss = false
const noise = true                  # Add random noise to fake A law
rng_seed() = MersenneTwister(1234)   # Random seed

# Machine learning training
current_epoch = 1
loss_history = []