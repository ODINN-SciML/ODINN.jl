using LinearAlgebra
using Statistics
using Zygote
using PaddedViews
using Flux
using Flux: @epochs
using Tullio
using Plots
using Infiltrator

#### Parameters
nx, ny = 100, 100 # Size of the grid
Δx, Δy = 1, 1
Δt = 0.01
t₁ = 5

D₀ = 1
tolnl = 1e-4
itMax = 100
damp = 0.85
dτsc   = 1.0/3.0
ϵ     = 1e-4            # small number
cfl  = max(Δx^2,Δy^2)/4.1

A₀ = 1
ρ = 9
g = 9.81
n = 3
p = (Δx, Δy, Δt, t₁, ρ, g, n)  # we add extra parameters for the nonlinear diffusivity

### Reference dataset for the heat Equations
T₀ = [ 250 * exp( - ( (i - nx/2)^2 + (j - ny/2)^2 ) / 300 ) for i in 1:nx, j in 1:ny ];
T₁ = copy(T₀);

#######   FUNCTIONS   ############

# Utility functions
@views avg(A) = 0.25 * ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )

@views avg_x(A) = 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )

@views avg_y(A) = 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )

### Functions to generate reference dataset to train UDE

function Heat_nonlinear(T, A, p)
   
    Δx, Δy, Δt, t₁, ρ, g, n = p
    
    #### NEW CODE TO BREAK
    dTdx = diff(T, dims=1) / Δx
    dTdy = diff(T, dims=2) / Δy
    ∇T = sqrt.(avg_y(dTdx).^2 .+ avg_x(dTdy).^2)

    D = A .* avg(T) .* ∇T
    # D = A

    dTdx_edges = diff(T[:,2:end - 1], dims=1) / Δx
    dTdy_edges = diff(T[2:end - 1,:], dims=2) / Δy
   
    Fx = -avg_y(D) .* dTdx_edges
    Fy = -avg_x(D) .* dTdy_edges 
    
    # Fx = -D * dTdx_edges
    # Fy = -D * dTdy_edges  
    
    F = .-(diff(Fx, dims=1) / Δx .+ diff(Fy, dims=2) / Δy)   

    # dτ = dτsc * min( 10.0 , 1.0/(1.0/Δt + 1.0/(cfl/(ϵ + D))))

    dτ = dτsc * min.( 10.0 , 1.0./(1.0/Δt .+ 1.0./(cfl./(ϵ .+ avg(D)))))
    
    return F, dτ
 
end

# Fake law to create reference dataset and to be learnt by the NN
fakeA(t) = A₀ * exp(2t)
# fakeA(t) = 0.1 * (2 + t^5 / t₁^4)

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
        err = Inf 
       
        while iter < itMax+1 && tol <= err
            
            Err = copy(T)

            if fake
                A = fakeA(t)  # compute the fake A value involved in the nonlinear diffusivity
            else
                # Compute A with the NN once per time step
                A = UA([t]')[1]  # compute A parameter involved in the diffusivity
            end
            
            F, dτ = Heat_nonlinear(T, A, p)

            @tullio ResT[i,j] := -(T[i,j] - Hold[i,j])/Δt + F[pad(i-1,1,1),pad(j-1,1,1)] 
            
            dTdt_ = copy(dTdt)
            @tullio dTdt[i,j] := dTdt_[i,j]*damp + ResT[i,j]

            T_ = copy(T)
            # @tullio T[i,j] := max(0.0, T_[i,j] + dτ * dTdt[i,j])
            @tullio T[i,j] := max(0.0, T_[i,j] + dTdt[i,j]*dτ[pad(i-1,1,1),pad(j-1,1,1)])
            
            Zygote.ignore() do
                Err .= Err .- T
                err = maximum(Err)
            end 
            
            iter += 1
            total_iter += 1
            
        end
        
        t += Δt
        
    end

    # if(!fake)
    #     println("Values of UA in heatflow_nonlinear: ", UA([0., .5, 1.]')) # Simulations here are correct
    # end
    
    return T
    
end

# Patch suggested by Michael Abbott needed in order to correctly retrieve gradients
Flux.Optimise.update!(opt, x::AbstractMatrix, Δ::AbstractVector) = Flux.Optimise.update!(opt, x, reshape(Δ, size(x)))

function train(loss, UA, p)
    
    @epochs 5 hybrid_train_NN!(loss, UA, p, opt, losses)
    
    # println("Values of UA in train(): ", UA([0., .5, 1.]'))
    
    return UA, losses
    
end

function hybrid_train_NN!(loss, UA, p, opt, losses)
    
    T = T₀
    θ = Flux.params(UA)
    # println("Values of UA in hybrid_train BEFORE: ", UA([0., .5, 1.]'))
    loss_UA, back_UA = Zygote.pullback(() -> loss(T, p), θ)
    push!(losses, loss_UA)
   
    ∇_UA = back_UA(one(loss_UA))

    println("Loss: ", loss_UA)

    # for ps in θ
    #    println("Gradients ∇_UA[ps]: ", ∇_UA[ps])
    # end
    
    # println("θ: ", θ) # parameters are NOT NaNs
    println("Values of UA in hybrid_train AFTER: ", UA([0., .5, 1.]')) # Simulations here are all NaNs
    
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
    Dense(10,10, leakyrelu, initb = Flux.glorot_normal), 
    Dense(10,5, leakyrelu, initb = Flux.glorot_normal), 
    Dense(5,1) 
)

opt = RMSProp()
losses = []

# Train heat equation UDE
UA_trained, losses = train(loss_NN, UA, p) 


all_times = LinRange(0, t₁, 1000)
# println("UA_trained(all_times')': ",  UA_trained(all_times')')
plot(all_times, UA_trained(all_times')', title="Simulated A values by the NN", yaxis="A", xaxis="Time", label="NN")
plot!(fakeA, 0, t₁, label="fake")