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
# minA = 1.58e-17
# maxA = 1.58e-16
minA = 1.58 # units changed to avoid numerical issues
maxA = 15.8 # to be multiplied by e-17 at the end

# 
ρ = 900                     # Ice density [kg / m^3]
g = 9.81                    # Gravitational acceleration [m / s^2]
n = 3                       # Glen's flow law exponent

α = 0                       # Weertman-type basal sliding (Weertman, 1964, 1972). 1 -> sliding / 0 -> no sliding
C = 15e-14                  # Sliding factor, between (0 - 25) [m⁸ N⁻³ a⁻¹]

Γ = (n-1) * (ρ * g)^n / (n+2) # 1 / m^3 s

### Differential equations ###
# Configuration of the forward model

# Model 
model = "standard"         # options: "standard", "fake A", "fake C" 
# Method to solve the DE
method = "implicit"        # options: implicit, explicit


η = 0.3                    # Parameter that control the stepsize of the numerical method. eta must be < 1
damp = 0.85
itMax = 100                # maximum number of iterations used in non-adaptive semi-implicit method
itMax_ref = 300            # maximum number of iterations used for genereting reference dataset
nout = 5                   # error check frequency
tolnl = 1e-1               # tolerance of semi-implicit method 
tolnl_ref  = 1e-3          # tolerance of semi-implicit method used to generate reference dataset
dτsc   = 1.0/3.0           # iterative dtau scaling
ϵ     = 1e-4               # small number
Δx = Δy = 50               # [m]
cfl  = max(Δx^2,Δy^2)/4.1

# Time 
t = 0                      # initial time
Δt = 1.0/12.0              # time step [yr]
Δts = []
t₁ = 10                     # number of simulation years 

## Climate parameters
base_url = ("https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/elev_bands") # OGGM elevation bands
mb_type = "mb_real_daily"
grad_type = "var_an_cycle" # could use here as well 'cte'
# fs = "_daily_".*climate
fs = "_daily_W5E5"

### Workflow ###
# var_format = "scalar"    # data format for the parameter to be learnt
var_format = "matrix"
create_ref_dataset = false
train_UDE = true