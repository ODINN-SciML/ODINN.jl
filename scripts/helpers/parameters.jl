####################################################
###   Global parameters for glacier simulations  ###
####################################################

### Physics  ###
# Ice diffusivity factor
#A = 2e-16   # varying factor (0.125 - 10)

# A ranging from 0.125 to 5
#A = 0.5e-24 #2e-16  1 / Pa^3 s
#A = 5e-24 #2e-16  1 / Pa^3 s
A = 1.3e-24 #2e-16  1 / Pa^3 s
A *= 60 * 60 * 24 * 365.25 # [1 / Pa^3 yr]

# 
const ρ = 900                     # Ice density [kg / m^3]
const g = 9.81                    # Gravitational acceleration [m / s^2]
const n = 3                       # Glen's flow law exponent

const α = 0                       # Weertman-type basal sliding (Weertman, 1964, 1972). 1 -> sliding / 0 -> no sliding
const C = 15e-14                  # Sliding factor, between (0 - 25) [m⁸ N⁻³ a⁻¹]

const Γ = (n-1) * (ρ * g)^n / (n+2) # 1 / m^3 s

### Differential equations ###
# Configuration of the forward model

# Model 
model = "standard"         # options: "standard", "fake A", "fake C" 
# Method to solve the DE
method = "implicit"        # options: implicit, explicit


const η = 0.3                    # Parameter that control the stepsize of the numerical method. eta must be < 1
const damp = 0.95                # Tuning parameter
const itMax = 500                # maximum number of iterations used in non-adaptive semi-implicit method
const itMax_ref = 500            # maximum number of iterations used for genereting reference dataset
const nout = 5                   # error check frequency
const tolnl = 1e-3               # tolerance of semi-implicit method 
const tolnl_ref  = 1e-3          # tolerance of semi-implicit method used to generate reference dataset
const dτsc   = 0.65              # Tuning parameter - iterative dtau scaling
const ϵ     = 1e-4               # small number
const Δx = 50                    # [m]
const Δy = 50
const cfl  = max(Δx^2,Δy^2)/4.1

# Time 
t = 0                      # initial time
const Δt = 1.0/12.0     
# Δt = 1.0/365.25          # time step [yr]
Δts = []
const t₁ = 2                     # number of simulation years 

## Climate parameters
const base_url = ("https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/elev_bands") # OGGM elevation bands
const mb_type = "mb_real_daily"
const grad_type = "var_an_cycle" # could use here as well 'cte'
# fs = "_daily_".*climate
const fs = "_daily_W5E5"

### Workflow ###
# var_format = "scalar"    # data format for the parameter to be learnt
var_format = "matrix"
USE_GPU = false # switch here to use GPU

create_ref_dataset = true  
train_UDE = true