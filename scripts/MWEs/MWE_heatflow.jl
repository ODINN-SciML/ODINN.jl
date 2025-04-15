using LinearAlgebra
using Statistics
using Zygote
using PaddedViews
using Flux
using Flux: @epochs
using Tullio
using Plots
using Infiltrator

using HDF5
using JLD

#### Parameters
nx, ny = 100, 100 # Size of the grid
Δx, Δy = 1, 1
Δt = 0.01
t₁ = 5

D₀ = 1
tolnl = 1e-4
itMax = 100
damp = 0.1
dτsc   = 1.0/3.0
ϵ     = 1e-4            # small number
cfl  = max(Δx^2,Δy^2)/4.1

A₀ = 1
ρ = 9
g = 9.81
n = 3
p = (Δx, Δy, Δt, t₁, ρ, g, n)  # we add extra parameters for the nonlinear diffusivity

#dataset = 'fake'
dataset = "Argentiere"

if dataset == "fake"

    ### Reference dataset for the heat Equations
    T₀ = [ 250 * exp( - ( (i - nx/2)^2 + (j - ny/2)^2 ) / 300 ) for i in 1:nx, j in 1:ny ];
    T₁ = copy(T₀);
    
elseif dataset == "Argentiere"
    
    root_dir = cd(pwd, ".")
    argentiere_f = h5open(joinpath(root_dir, "data/Argentiere_2003-2100_aflow2e-16_50mres_rcp2.6.h5"), "r")

    mutable struct Glacier
        bed::Array{Float64}    # bedrock height
        thick::Array{Float64}  # ice thickness
        vel::Array{Float64}    # surface velocities
        MB::Array{Float64}     # surface mass balance
        lat::Float64
        lon::Float64
    end
    
    # Fill the Glacier structure with the retrieved data
    argentiere = Glacier(HDF5.read(argentiere_f["bed"])[begin:end-2,:],
                         HDF5.read(argentiere_f["thick_hist"])[begin:end-2,:,2:end],
                         HDF5.read(argentiere_f["vel_hist"])[begin:end-2,:,2:end],
                         HDF5.read(argentiere_f["s_apply_hist"])[begin:end-2,:,2:end],
                         0, 0);
    
    nx = size(argentiere.bed)[1]
    ny = size(argentiere.bed)[2]
    
    T₀ = argentiere.thick[:,:,1]
    # B  = copy(argentiere.bed)
    
end

#######   FUNCTIONS   ############

# Utility functions
@views avg(A) = 0.25 * ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )

@views avg_x(A) = 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )

@views avg_y(A) = 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )

### Functions to generate reference dataset to train UDE

function Heat_nonlinear(T, A, p)
   
    Δx, Δy, Δt, t₁, ρ, g, n = p
    
    #### NEW CODE TO BREAK  ########

    dTdx = diff(T, dims=1) / Δx
    dTdy = diff(T, dims=2) / Δy
    ∇T = sqrt.(avg_y(dTdx).^2 .+ avg_x(dTdy).^2)

    # D = A * avg(T) .* ∇T # breaking

    Γ = 2 * A * (ρ * g)^n / (n+2)
    D = Γ .* avg(T).^(n + 2) .* ∇T.^(n - 1) # test with SIA 
    # D = A

    dTdx_edges = diff(T[:,2:end - 1], dims=1) / Δx
    dTdy_edges = diff(T[2:end - 1,:], dims=2) / Δy
   
    Fx = -avg_y(D) .* dTdx_edges # breaking
    Fy = -avg_x(D) .* dTdy_edges 
    
    # Fx = -D * dTdx_edges
    # Fy = -D * dTdy_edges  
    
    F = .-(diff(Fx, dims=1) / Δx .+ diff(Fy, dims=2) / Δy)   

    # dτ = dτsc * min( 10.0 , 1.0/(1.0/Δt + 1.0/(cfl/(ϵ + D))))

    dτ = dτsc * min.( 10.0 , 1.0./(1.0/Δt .+ 1.0./(cfl./(ϵ .+ avg(D))))) # breaking

    #########################
    
    return F, dτ
 
end

# Fake law to create reference dataset and to be learnt by the NN
# fakeA(t) = A₀ * exp(2t)
# fakeA(t) = 0.1 * (2 + t^5 / t₁^4)


# fakeA(t) = (t + t^2) .* 2e-16 # original

# predict_A(UA, t) = (0 .+ UA(t)).*2e-16 # For the base value I've been trying 
                              # either the avg value or the initial (t=0) value

predict_A(UA, t) = (2 .+ UA(t)).*1e-16 # SIA

fakeA(t) = (1 + t^1.7)*1e-17

### Heat equation based on a fake A parameter function to compute the diffusivity
function heatflow_nonlinear(T, p, fake, tol=Inf)
   
    Δx, Δy, Δt, t₁, ρ, g, n = p
    
    total_iter = 0
    t = 0
    
    while t < t₁
        
        iter = 1
        err = 2 * tolnl
        Hold = copy(T)
        dTdt = zeros(nx, ny)
        # err = Inf 
       
        while iter < itMax+1 && tolnl <= err
            
            Err = copy(T)

            if fake
                A = fakeA(t)  # compute the fake A value involved in the nonlinear diffusivity
            else
                # Compute A with the NN once per time step
                A = predict_A(UA, [t]')[1]  # compute A parameter involved in the diffusivity
            end
            
            F, dτ = Heat_nonlinear(T, A, p)

            @tullio ResT[i,j] := -(T[i,j] - Hold[i,j])/Δt + F[pad(i-1,1,1),pad(j-1,1,1)] 
            
            dTdt_ = copy(dTdt)
            @tullio dTdt[i,j] := dTdt_[i,j]*damp + ResT[i,j]

            T_ = copy(T)
            # @tullio T[i,j] := max(0.0, T_[i,j] + dτ * dTdt[i,j])
            @tullio T[i,j] := max(0.0, T_[i,j] + dTdt[i,j]*dτ[pad(i-1,1,1),pad(j-1,1,1)])  # breaking
            
            Zygote.ignore() do
                Err .= Err .- T
                err = maximum(Err)
                # println("error at iter ", iter, ": ", err)

            end 
            
            iter += 1
            total_iter += 1
            
        end
        
        # println("t: ", t)
        t += Δt
        
    end

    # if(!fake)
    #     println("Values of UA in heatflow_nonlinear: ", UA([0., .5, 1.]')) # Simulations here are correct
    # end
    
    return T
    
end

# Patch suggested by Michael Abbott needed in order to correctly retrieve gradients
Flux.Optimise.update!(opt, x::AbstractMatrix, Δ::AbstractVector) = Flux.Optimise.update!(opt, x, reshape(Δ, size(x)))

function train(loss, UA, T, p)
    
    @epochs 1 hybrid_train_NN!(loss, UA, T, p, opt, losses)
    
    # println("Values of UA in train(): ", UA([0., .5, 1.]'))
    
    return UA, losses
    
end

function hybrid_train_NN!(loss, UA, T, p, opt, losses)
    
    θ = Flux.params(UA)
    # println("Values of UA in hybrid_train BEFORE: ", UA([0., .5, 1.]'))
    loss_UA, back_UA = Zygote.pullback(() -> loss(T, p), θ)
    push!(losses, loss_UA)
   
    ∇_UA = back_UA(one(loss_UA))

    println("Loss: ", loss_UA)

    for ps in θ
        println("ps: ", ps)
       println("Gradients ∇_UA[ps]: ", ∇_UA[ps])
    end
    
    # println("θ: ", θ) # parameters are NOT NaNs
    println("Values of predict_A in hybrid_train AFTER: ", predict_A(UA, [1,2,3,4,5]')) # Simulations here are all NaNs
    
    Flux.Optimise.update!(opt, θ, ∇_UA)
    
end


function loss_NN(T, p, λ=1)

    T = heatflow_nonlinear(T, p, false)

    Zygote.ignore() do
        # println("Values of UA in loss_NN: ", UA([0., .5, 1.]')) # Simulations here are all NaNs
        display(heatmap(T - T_ref, clim=(0, maximum(T₀)), title="Error"))
    end
    l_cost = sqrt(Flux.Losses.mse(T, T_ref; agg=mean))

    return l_cost 
end


#######################

########################################
#####  TRAIN 2D HEAT EQUATION PDE  #####
########################################

T₂ = copy(T₀)
# Reference temperature dataset
T_ref = heatflow_nonlinear(T₂, p, true, 1e-1)

# display(heatmap(T₀ - T_ref, clim=(0, maximum(T₀)), title="T₀"))
# display(heatmap(T_ref, clim=(0, maximum(T₀)), title="T_ref"))

leakyrelu(x, a=0.01) = max(a*x, x)
relu(x) = max(0, x)

UA = Chain(
    Dense(1,10), 
    Dense(10,10, leakyrelu, init = Flux.glorot_normal), 
    Dense(10,10, leakyrelu, init = Flux.glorot_normal), 
    Dense(10,5, leakyrelu, init = Flux.glorot_normal), 
    Dense(5,1) 
)

# UA = Chain(
#     Dense(1,10), 
#     Dense(10,10, leakyrelu, init = Flux.kaiming_normal(gain=5)), 
#     Dense(10,10, leakyrelu, init = Flux.kaiming_normal(gain=5)), 
#     Dense(10,5, leakyrelu, init = Flux.kaiming_normal(gain=5)), 
#     Dense(5,1) 
# )

opt = RMSProp(0.001)
losses = []

# Train heat equation UDE
T = copy(T₀)
# UA_trained, losses = train(loss_NN, UA, T, p) 

hybrid_train_NN!(loss_NN, UA, T, p, opt, losses)


all_times = LinRange(0, t₁, 50)
# println("UA_trained(all_times')': ",  UA_trained(all_times')')
plot(all_times, predict_A(UA_trained, all_times')', title="Simulated A values by the NN", yaxis=:A, xaxis="Time", label="NN")
plot!(fakeA, 0, t₁, label="fake")