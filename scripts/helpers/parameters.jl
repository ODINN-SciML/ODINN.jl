####################################################
###   Global parameters for glacier simulations  ###
####################################################

### Physics  ###
# Ice diffusivity factor
#A = 2e-16   # varying factor (0.125 - 10)

# A ranging from 0.125 to 5
#A = 0.5e-24 #2e-16  1 / Pa^3 s
A = 5e-24 #2e-16  1 / Pa^3 s
# A = 1.3e-24 #2e-16  1 / Pa^3 s
A *= 60 * 60 * 24 * 365.25 # 1 / Pa^3 yr

# Ice density
ρ = 900 # kg / m^3
# Gravitational acceleration
g = 9.81 # m / s^2
# Glen's flow law exponent
n = 3

# Weertman-type basal sliding (Weertman, 1964, 1972) 
α = 1   # 1 -> sliding / 0 -> no sliding
C = 15e-14 # m⁸ N⁻³ a⁻¹   Sliding factor, between (0 - 25)

Γ = (n-1) * (ρ * g)^n / (n+2) # 1 / m^3 s

### Differential equations ###
# Configuration of the forward model

# Model 
model = "fake A" # options are: "standard", "fake A", "fake C" 
# Method to solve the DE
method = "explicit-adaptive" #"explicit"
#method = "explicit" 

# Parameter that control the stepsize of the numerical method 
# η < 1 is requiered for stability
η = 0.9
#η = 0.2   

# Time 
#Δt = 0.001 # requiered for "explicit" / It need to be added to p
t = 0
Δts = []
t₁ = 2 # number of simulation years 

### Workflow ###
create_ref_dataset = true
train_UDE = true