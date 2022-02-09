const t₁ = 5                      # number of simulation years 
const ρ = 900                     # Ice density [kg / m^3]
const g = 9.81                    # Gravitational acceleration [m / s^2]
const n = 3                       # Glen's flow law exponent
const maxA = 8e-16
const minA = 3e-17
const maxT = 1
const minT = -25

create_ref_dataset = false
const noise = true # Add random noise to fake A law

const minA_out = 0.3
const maxA_out = 8
