## Environment and packages
using Statistics
using LinearAlgebra
using Random 
using OrdinaryDiffEq
using DiffEqFlux
using Flux
using Distributed
using Tullio
using RecursiveArrayTools
using Infiltrator
using Plots

const t₁ = 5                 # number of simulation years 
const ρ = 900                     # Ice density [kg / m^3]
const g = 9.81                    # Gravitational acceleration [m / s^2]
const n = 3                      # Glen's flow law exponent
const maxA = 8e-16
const minA = 3e-17
const maxT = 1
const minT = -25
A = 1.3e-24 #2e-16  1 / Pa^3 s
A *= Float64(60 * 60 * 24 * 365.25) # [1 / Pa^3 yr]
C = 0
α = 0

@views avg(A) = 0.25 .* ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )
@views avg_x(A) = 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )
@views avg_y(A) = 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )
@views diff_x(A) = (A[begin + 1:end, :] .- A[1:end - 1, :])
@views diff_y(A) = (A[:, begin + 1:end] .- A[:, 1:end - 1])
@views inn(A) = A[2:end-1,2:end-1]


callback = function (θ,l) # callback function to observe training
    @show l
    false
  end

function loss_iceflow(θ, context, UA) 

    H = context[2]
    tspan = (0.0,t₁)

    iceflow_UDE!(dH, H, θ, t) = iceflow_NN!(dH, H, θ, t, context, temp_series[5], UA) # closure
    iceflow_prob = ODEProblem(iceflow_UDE!,H,tspan,θ)
    H_pred = solve(iceflow_prob, BS3(), u0=H, p=θ, reltol=1e-6, 
                    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()), save_everystep=false, 
                    progress=true, progress_steps = 10)


    # Compute loss function for the full batch
    l_H = sum(H_pred.u[end])
    
    return l_H
end

function iceflow_NN!(dH, H, θ, t, context, temps, UA)
    # ArrayPartition(B, H, current_year, temp_series, batch_idx) 

    year = floor(Int, t) + 1
    if year <= t₁
        temp = temps[year]
    else
        temp = temps[year-1]
    end

    A = predict_A̅(UA, θ, [temp]) # FastChain prediction requires explicit parameters

    # Compute the Shallow Ice Approximation in a staggered grid
    dH .= SIA(dH, H, A, context)
    # println("dH: ", maximum(dH))
end  


# Function without mutation for Zygote, with context as an ArrayPartition
function SIA(dH, H, A, context)
    # Retrieve parameters
    B = context[1]

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n - 1)/2) 

    Γ = 2 * A * (ρ * g)^n / (n+2) # 1 / m^3 s 
    D = Γ .* avg(H).^(n + 2) .* ∇S

    # Compute flux components
    dSdx_edges = diff_x(S[:,2:end - 1]) / Δx
    dSdy_edges = diff_y(S[2:end - 1,:]) / Δy
    Fx = .-avg_y(D) .* dSdx_edges
    Fy = .-avg_x(D) .* dSdy_edges 

    #  Flux divergence
    @tullio dH[i,j] := -(diff_x(Fx)[pad(i-1,1,1),pad(j-1,1,1)] / Δx + diff_y(Fy)[pad(i-1,1,1),pad(j-1,1,1)] / Δy) # MB to be added here 

    return dH
end


function A_fake(temp)
    return @. minA + (maxA - minA) * ((temp-minT)/(maxT-minT) )^2
end

predict_A̅(UA, θ, temp) = UA(temp, θ) .* 1e-16

function fake_temp_series(t, means=Array{Float64}([0,-2.0,-3.0,-5.0,-10.0,-12.0,-14.0,-15.0,-20.0]))
    temps, norm_temps, norm_temps_flat = [],[],[]
    for mean in means
       push!(temps, mean .+ rand(t).*1e-1) # static
       append!(norm_temps_flat, mean .+ rand(t).*1e-1) # static
    end

    # Normalise temperature series
    norm_temps_flat = Flux.normalise([norm_temps_flat...]) # requires splatting

    # Re-create array of arrays 
    for i in 1:t₁:length(norm_temps_flat)
        push!(norm_temps, norm_temps_flat[i:i+(t₁-1)])
    end

    return temps, norm_temps
end

##################################################
#### Generate reference dataset ####
##################################################

nx = ny = 100
const B = zeros(Float64, (nx, ny))
const σ = 1000
H₀ = Matrix{Float64}([ 250 * exp( - ( (i - nx/2)^2 + (j - ny/2)^2 ) / σ ) for i in 1:nx, j in 1:ny ])    
Δx = Δy = 50 #m   
# ensemble = EnsembleSerial()
ensemble = EnsembleDistributed()
const minA_out = 0.3
const maxA_out = 8
sigmoid_A(x) = minA_out + (maxA_out - minA_out) / ( 1 + exp(-x) )

const temp_series, norm_temp_series = fake_temp_series(t₁)

# Train UDE
UA = FastChain(
        FastDense(1,3, x->tanh.(x)),
        FastDense(3,10, x->tanh.(x)),
        FastDense(10,3, x->tanh.(x)),
        FastDense(3,1, sigmoid_A)
    )
θ = initial_params(UA)


H = deepcopy(H₀)
current_year = 0f0
# Tuple with all the temp series and H_refs
context = (B, H, current_year, temp_series)
loss(θ) = loss_iceflow(θ, context, UA) # closure

# Debugging
println("Gradients: ", gradient(loss, θ))

