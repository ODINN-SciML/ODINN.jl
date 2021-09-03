using HDF5
using JLD
using LinearAlgebra
using Statistics
using Zygote
using PaddedViews
using Flux
using Flux: @epochs
using Tullio
using Plots
using Infiltrator

### Global parameters  ###
include("../helpers/parameters.jl")
### Types  ###
include("../helpers/types.jl")

include("../helpers/utils.jl")

#### Parameters
nx, ny = 100, 100 # Size of the grid
Δx, Δy = 1, 1
Δt = 1.0/12.0
t = 0
t₁ = 2

D₀ = 1
tolnl = 1e-4
itMax = 100
damp = 0.85
dτsc   = 1.0/3.0
ϵ     = 1e-4            # small number
cfl  = max(Δx^2,Δy^2)/4.1

method = "implicit"

A₀ = 1
ρ = 9
g = 9.81
n = 3

### Reference dataset for the heat Equations

# Load the HDF5 file with Harry's simulated data
root_dir = cd(pwd, ".")
argentiere_f = h5open(joinpath(root_dir, "data/Argentiere_2003-2100_aflow2e-16_50mres_rcp2.6.h5"), "r")

# Fill the Glacier structure with the retrieved data
argentiere = Glacier(HDF5.read(argentiere_f["bed"])[begin:end-2,:],
                     HDF5.read(argentiere_f["thick_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["vel_hist"])[begin:end-2,:,2:end],
                     HDF5.read(argentiere_f["s_apply_hist"])[begin:end-2,:,2:end],
                     0, 0)

# Update mass balance data with NaNs
MB_plot = copy(argentiere.MB)
voidfill!(MB_plot, argentiere.MB[1,1,1])

# Domain size
nx = size(argentiere.bed)[1]
ny = size(argentiere.bed)[2];

B  = copy(argentiere.bed)
H₀ = copy(argentiere.thick[:,:,1])
v = zeros(size(argentiere.thick)) # surface velocities

# Spatial and temporal differentials
Δx = Δy = 50 #m (Δx = Δy)

MB_avg = []
for year in 1:length(argentiere.MB[1,1,:])
    MB_buff = buffer_mean(argentiere.MB, year)
    voidfill!(MB_buff, MB_buff[1,1], 0)
    push!(MB_avg, MB_buff)
end 


# H₀ = [ 250 * exp( - ( (i - nx/2)^2 + (j - ny/2)^2 ) / 300 ) for i in 1:nx, j in 1:ny ];
H₁ = copy(H₀);

#######   FUNCTIONS   ############

# Utility functions
@views avg(A) = 0.25 * ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )

@views avg_x(A) = 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )

@views avg_y(A) = 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )

### Functions to generate reference dataset to train UDE

function iceflow!(H,H_ref::Dict, A, p,t,t₁)

    println("Running forward PDE ice flow model...\n")
    # Instantiate variables
    let             
    current_year = 0
    total_iter = 0
    ts_i = 1

    # Manual explicit forward scheme implementation
    while t < t₁
        let
        iter = 1
        err = 2 * tolnl
        Hold = copy(H) # hold value of H for the other iteration in the implicit method
        # we need to define dHdt for iter = 1
        #dHdt = zeros(nx-2, ny-2) # with broadcasting
        dHdt = zeros(nx, ny) # with Tullio

        # Get current year for MB and ELA
        year = floor(Int, t) + 1
        # if(year != current_year)
            
        #     # Predict A with the fake A law
        #     ŶA = A_fake(MB_avg[year], size(H))

        #     Zygote.ignore() do
        #         if(year == 1)
        #             println("ŶA max: ", maximum(ŶA))
        #             println("ŶA min: ", minimum(ŶA))
        #         end

        #     
        #     # Unpack and repack tuple with updated A value
        #     Δx, Δy, Γ, A, B, v, MB, MB_avg, C, α = p
        #     p = (Δx, Δy, Γ, ŶA, B, v, MB, MB_avg, C, α)
        #     current_year = year
        #     println("Year ", year)
        # end
        y = (year, current_year)

        if method == "explicit"

            F, dτ, current_year = SIA(H, p, y)
            inn(H) .= max.(0.0, inn(H) .+ Δt * F)
            t += Δt_exp
            total_iter += 1 

        elseif method == "implicit"

            #while err > tolnl && iter < itMax+1
            while iter < itMax + 1
            
                Err = copy(H)

                # Compute the Shallow Ice Approximation in a staggered grid
                
                F, dτ, current_year = SIA(H, p, y)

                # Implicit method with broadcasting
                # Differentiate H via a Picard iteration method
                #ResH = -(inn(H) .- inn(Hold))/Δt .+ F  # with broadcasting
                #dHdt = damp .* dHdt .+ ResH # with broadcasting
                # Update the ice thickness
                #inn(H) .= max.(0.0, inn(H) .+ dτ .* dHdt) # with broadcasting 

                # implicit method with Tullio  
                @tullio ResH[i,j] := -(H[i,j] - Hold[i,j])/Δt + F[pad(i-1,1,1),pad(j-1,1,1)]
                
                dHdt_ = copy(dHdt)
                @tullio dHdt[i,j] := dHdt_[i,j]*damp + ResH[i,j]
                
                H_ = copy(H)
                @tullio H[i,j] := max(0.0, H_[i,j] + dHdt[i,j]*dτ[pad(i-1,1,1),pad(j-1,1,1)])
                

                if mod(iter, nout) == 0
                    # Compute error for implicit method with damping
                    Err = Err .- H
                    err = maximum(Err)
                    # println("error at iter ", iter, ": ", err)

                    if isnan(err)
                        error("""NaNs encountered.  Try a combination of:
                                    decreasing `damp` and/or `dtausc`, more smoothing steps""")
                    end
                end
            
                iter += 1
                total_iter += 1
            end

            t += Δt

        end
        end

        # Store timestamps to be used for training of the UDEs
        if ts_i < length(H_ref["timestamps"])+1
            if t >= H_ref["timestamps"][ts_i]
                println("Saving H at year ", H_ref["timestamps"][ts_i])
                push!(H_ref["H"], H)
                ts_i += 1
            end          
        end        
    end 

    println("Total Number of iterartions: ", total_iter)
    end
    
    println("Saving reference data")
    save(joinpath(root_dir, "data/H_ref.jld"), "H", H_ref)

    return H
end

function iceflow!(H, UA, p,t,t₁, inverse)

    # Retrieve input variables  
    let                  
    current_year = 0
    total_iter = 0
    global model = "UDE_A"
    MB_avg = p[8] 

    # Forward scheme implementation
    while t < t₁
        let
        iter = 1
        err = 2 * tolnl
        Hold = copy(H)
        dHdt = zeros(nx, ny)

        # Get current year for MB and ELA
        year = floor(Int, t) + 1

        if(year != current_year) # only for UDE problem
            current_year = year
            println("Year ", year)

            if(!inverse)
            
                ## Predict A with the NN
                # ŶA = UA(vec(MB_avg[year])') .* 1e-17 # Adding units outside the NN
                ## Scalar version            
                YA = UA([mean(vec(MB_avg[year])')])[1] .* 1e-17 # Adding units outside the NN

                Zygote.ignore() do
                    # println("Current params: ", Flux.params(UA))

                    println("YA: ", YA )

                    #display(heatmap(MB_avg[year], title="MB"))
                end
            
                ## Unpack and repack tuple with updated A value
                Δx, Δy, Γ, A, B, v, MB, MB_avg, C, α = p
                p = (Δx, Δy, Γ, YA, B, v, MB, MB_avg, C, α)
            
            else
                A = UA
            end
        end
        y = (year, current_year)

        if method == "implicit"
            
            #while err > tolnl && iter < itMax+1
            while iter < itMax + 1

                #println("iter: ", iter)
            
                Err = copy(H)

                # Compute the Shallow Ice Approximation in a staggered grid
                F, dτ, current_year = SIA(H, A, p, y)

                # Compute the residual ice thickness for the inertia
                @tullio ResH[i,j] := -(H[i,j] - Hold[i,j])/Δt + F[pad(i-1,1,1),pad(j-1,1,1)]

                dHdt_ = copy(dHdt)
                @tullio dHdt[i,j] := dHdt_[i,j]*damp + ResH[i,j]
                              
                # We keep local copies for tullio
                H_ = copy(H)
                
                # Update the ice thickness
                @tullio H[i,j] := max(0.0, H_[i,j] + dHdt[i,j]*dτ[pad(i-1,1,1),pad(j-1,1,1)])

                #println("maximum H: ",maximum(H))
                #println("maximum H on borders: ", maximum([maximum(H[1,:]), maximum(H[:,1]), maximum(H[nx,:]), maximum(H[:,ny])]))

                #@show isderiving()
              
                Zygote.ignore() do
                    if mod(iter, nout) == 0
                        # Compute error for implicit method with damping
                        Err = Err .- H
                        err = maximum(Err)
                        # println("error: ", err)
                        #@infiltrate

                        if isnan(err)
                            error("""NaNs encountered.  Try a combination of:
                                        decreasing `damp` and/or `dtausc`, more smoothing steps""")
                        end
                    end
                end

                iter += 1
                total_iter += 1

            end

            #println("t: ", t)
          
            t += Δt
        end
        end # let
    end   
    end # let


    return H

end

function SIA(H, A, p, y)
    Δx, Δy, Γ, A_def, B, v, argentiere.MB, MB_avg, C, α = p
    year, current_year = y

    # Update glacier surface altimetry
    # S = B .+ H
    S = H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx  = diff(S, dims=1) / Δx
    dSdy  = diff(S, dims=2) / Δy
    ∇S = sqrt.(avg_y(dSdx).^2 .+ avg_x(dSdy).^2)

    # Compute diffusivity on secondary nodes
    # A here should have the same shape as H
    #                                     ice creep  +  basal sliding
    #D = (avg(pad(H)).^n .* ∇S.^(n - 1)) .* (A.*(avg(pad(H))).^(n-1) .+ (α*(n+2)*C)/(n-2)) 
    # Γ = 2 * A * (ρ * g)^n / (n+2) # 1 / m^3 s # Γ from Jupyter notebook. To do: standarize this value in patameters.jl
    if(model == "standard")
        Γ = 2 * A * (ρ * g)^n / (n+2)
    elseif(model == "UDE_A")
        # Matrix version
        # Γ = 2 * avg(reshape(A, size(H))) * (ρ * g)^n / (n+2) # 1 / m^3 s # Γ from Jupyter notebook. To do: standarize this value in patameters.jl
        
        
        # Γ = 2 * avg(reshape(A, size(H))) * (ρ * g)^n / (n+2) # 1 / m^3 s # Γ from Jupyter notebook. To do: standarize this value in patameters.jl
       #Γ = 2 * A * (ρ * g)^n / (n+2)

       # Scalar version
    #    Zygote.ignore() do
    #         @infiltrate
    #    end
       Γ = 2 * A * (ρ * g)^n / (n+2) # 1 / m^3 s # Γ from Jupyter notebook. To do: standarize this value in patameters.jl

    end
    D = Γ .* avg(H).^(n + 2) .* ∇S.^(n - 1) 
  
    #D = (Γ * avg(H).^n .* ∇S.^(n - 1)) .* (avg(reshape(A, size(H))) .* avg(H)).^(n-1) .+ (α*(n+2)*C)/(n-2)

    # Compute flux components
    dSdx_edges = diff(S[:,2:end - 1], dims=1) / Δx
    dSdy_edges = diff(S[2:end - 1,:], dims=2) / Δy
    Fx = .-avg_y(D) .* dSdx_edges
    Fy = .-avg_x(D) .* dSdy_edges    
    #  Flux divergence
    F = .-(diff(Fx, dims=1) / Δx .+ diff(Fy, dims=2) / Δy) # MB to be added here 

    # Compute dτ for the implicit method
    dτ = dτsc * min.( 10.0 , 1.0./(1.0/Δt .+ 1.0./(cfl./(ϵ .+ avg(D)))))

    # Compute velocities
    # Vx = -D./(av(H) .+ epsi).*av_ya(dSdx)
    # Vy = -D./(av(H) .+ epsi).*av_xa(dSdy)

    return F, dτ, current_year
    #return sum(F)

end

# Patch suggested by Michael Abbott needed in order to correctly retrieve gradients
Flux.Optimise.update!(opt, x::AbstractMatrix, Δ::AbstractVector) = Flux.Optimise.update!(opt, x, reshape(Δ, size(x)))

function loss(H, UA, p, t, t₁, inverse)
    l_H, l_A  = 0.0, 0.0
   
    H = iceflow!(H, UA, p,t,t₁, inverse)

    # A = p[4]
    # l_A = max((A-20)*100, 0) + abs(min((A-1)*100, 0))
    l_H = sqrt(Flux.Losses.mse(H, H_ref["H"][end]; agg=mean))

    # println("l_A: ", l_A)
    println("l_H: ", l_H)

    # l = l_A + l_H

    Zygote.ignore() do
        if(!inverse)
            println("Values of UA in loss(): ", UA([0., .5, 1.]'))
        end

       hml = heatmap(H_ref["H"][end] .- H, title="Loss error")
       display(hml)
    end

    return l_H
end

function hybrid_train_NN!(loss, UA, p, opt, losses, inverse)
    
    H = H₀
    θ = Flux.params(UA)
    # println("Values of UA in hybrid_train BEFORE: ", UA([0., .5, 1.]'))

    if inverse
        loss_UA, back_UA = Zygote.pullback(A -> loss(H, A, p, t, t₁, inverse), A) # inverse problem
    else
        loss_UA, back_UA = Zygote.pullback(() -> loss(H, UA, p, t, t₁, inverse), θ) # with UA
    end

    push!(losses, loss_UA)

    # @infiltrate
   
    ∇_UA = back_UA(one(loss_UA))

    println("Loss: ", loss_UA)

    for ps in θ
       println("Gradients ∇_UA[ps]: ", ∇_UA[ps])
    end
    
    # println("θ: ", θ) # parameters are NOT NaNs
    
    Flux.Optimise.update!(opt, θ, ∇_UA)
    println("Values of UA in hybrid_train in hybrid_train(): ", UA([0., .5, 1.]')) # Simulations here are all NaNs

end

function train(loss, UA, p, inverse)

    if inverse
        println("Running inverse problem")
    else
        println("Running UDE problem")
    end
    
    @epochs 5 hybrid_train_NN!(loss, UA, p, opt, losses, inverse)
    
    println("Values of UA in train(): ", UA([0., .5, 1.]'))
    
end



#######################

####################################################
#####  TRAIN 2D SHALLOW ICE APPROXIMATION PDE  #####
####################################################

A = 2e-16
p = (Δx, Δy, Γ, A, B, v, argentiere.MB, MB_avg, C, α) 
## Reference temperature dataset
# H_ref = Dict("H"=>[], "timestamps"=>[1,2,3])
# H = iceflow!(H₀,H_ref,A,p,t,t₁)

H_ref = load(joinpath(root_dir, "data/H_ref.jld"))["H"]

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

# Train iceflow UDE
inverse = true
A = 5e-16
p = (Δx, Δy, Γ, A, B, v, argentiere.MB, MB_avg, C, α) 
train(loss, UA, p, inverse) 

all_times = LinRange(0, t₁, 1000)
# println("UD(all_times')': ",  UD_trained(all_times')')
plot(fakeA, 0, t₁, label="fake")
plot!(all_times, UA(all_times')', title="Simulated A values by the NN", yaxis="A", xaxis="Time", label="NN")
