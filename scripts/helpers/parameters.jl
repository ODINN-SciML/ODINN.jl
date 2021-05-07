####################################################
###   Global parameters for glacier simulations  ###
####################################################

### Physics  ###
# Ice diffusivity factor
A = 2e-16
# A = 1.3e-24 #2e-16  1 / Pa^3 s
# A *= 60 * 60 * 24 * 365.25 # 1 / Pa^3 yr

# Ice density
ρ = 900 # kg / m^3
# Gravitational acceleration
g = 9.81 # m / s^2
# Glen's flow law exponent
n = 3

Γ = 2 * A * (ρ * g)^n / (n+2) # 1 / m^3 s

### Differential equations ###
# Configuration of the forward model
# Parameter that control the stepsize of the numerical method 
# η < 1 is requiered for stability
η = 0.2;
# Time 
t = 0
Δts = []
t₁ = 25 # number of simulation years 