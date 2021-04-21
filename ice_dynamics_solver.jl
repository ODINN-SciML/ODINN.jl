#= Glacier ice dynamics toy model

Toy model based on the Shallow Ice Approximation (SIA), mixing
partial differential equations (PDEs), neural networks and model
interpretation using SINDy.
=#

## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using Infiltrator
#using Debugger

# using Flux, DiffEqFlux, DataDrivenDiffEq
# using Flux: @epochs
# using Zygote
using DifferentialEquations
using Plots
using Measures
gr()
using Base: @kwdef
using Statistics
using ModelingToolkit
using LinearAlgebra
using CartesianGrids
using HDF5
# Set a random seed for reproduceable behaviour
using Random

###############################################
############  TYPES      #####################
###############################################

@kwdef mutable struct Hyperparameters
    batchsize::Int = 500    # batch size
    η::Float64 = 0.1   # learning rate
    epochs::Int = 500        # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
end

mutable struct Glacier
    bed::Array{Float32}    # bedrock height
    thick::Array{Float32}  # ice thickness
    vel::Array{Float32}    # surface velocities
end

###############################################
############  FUNCTIONS   #####################
###############################################

# Average of a matrix in the x direction with padding
function avg_x(A)
    A̅ = 0.5.*(A[1:end-1,:].+A[2:end,:])
    A̅ = vcat(A̅, A̅[end,:]')
    A̅_pn = Nodes(Primal, A)
    A̅_pn .= A̅
    return A̅_pn
end

# Average of a matrix in the y direction with padding
function avg_y(A)
    A̅ = 0.5.*(A[:,1:end-1].+A[:,2:end])
    A̅ = hcat(A̅, A̅[:,end])
    A̅_pn = Nodes(Primal, A)
    A̅_pn .= A̅
    return A̅_pn
end

# 4 point average of Dual Nodes to a single Primal Node
function four_point_avg!(Nodes_4p_avg::Nodes{Primal}, dualNodes::Nodes{Dual}, i,j)
    Nodes_4p_avg[i,j] = (dualNodes[i,j] + dualNodes[i+1,j] + dualNodes[i,j+1] + dualNodes[i+1,j+1])/4
end

# Performs a 4 point average of Node values of the Dual grid to the Primal grid
function dual2primal_4p_avg!(primalNodes::Nodes{Primal}, dualNodes::Nodes{Dual})
    for i in 1:size(primalNodes)[1]
        for j in 1:size(primalNodes)[2]
            four_point_avg!(primalNodes, dualNodes, i,j)
        end
    end
end

# Norm of a gradient field
function ∇_norm(gradEdges::Edges{Dual}, dualNodes::Nodes{Dual})
    ∇_norm = Nodes(Primal, dualNodes)
    for i in 1:size(∇_norm)[1]
        for j in 1:size(∇_norm)[2]
            # 2-point mean of gradients
            u̅ = mean([gradEdges.u[i,j], gradEdges.u[i,j+1]])
            v̅ = mean([gradEdges.v[i,j], gradEdges.v[i+1,j]])
            # Norm
            ∇_norm[i,j] = norm([u̅, v̅])
        end
    end

    return ∇_norm
end

# Compute diffusivities on u and v
function diffusivity(vars, p, ref_dn)
    A, ρ, g, n = p
    H, ∇S_norm = vars

    D = ((2*A)/(n+2)).*((ρ*g.*H).^n).*(H.^2).*(∇S_norm.^(n-1))

    # Diffusivities Nodes
    D_pn = Nodes(Primal,ref_dn)
    D_pn .= D

    return D_pn
end

# Forward model with Δt timestepping
function SIA!(du,u,p,t)

    # Retrieve input variables
    H, S = u

    ####################################
    ##### Create staggered grid  #######
    ####################################
    # Ice thickness
    H_dn = Nodes(Dual,size(H))
    H_dn .= H

    # DEM
    S_dn = Nodes(Dual,size(S))
    S_dn .= S

    #####################################################
    #### 1 - Calculate diffusivities in Primal grid  ####
    #####################################################
    #println("Calculating diffusivities in Primal Nodes \n")

    # 1.1: 4-point average of ice thickness from Dual to Primal  
    #println("H_dn: ", H_dn.data[100,100])
    H_pn = Nodes(Primal, H_dn)
    dual2primal_4p_avg!(H_pn, H_dn)
    S_pn = Nodes(Primal,S_dn)
    dual2primal_4p_avg!(S_pn, S_dn)

    # 1.2: 2-point central estimate of surface gradient from Dual Nodes to Dual Edges
    ∇_S_de = grad(S_dn)/Δx

    # 1.3: 2-point average of squared norm of gradients from Dual Edges to Primal Nodes
    ∇S_norm = ∇_norm(∇_S_de, S_dn)

    # 1.4: Calculate diffusivities on Primal Nodes
    vars = H_pn.data, ∇S_norm.data
    D_pn = diffusivity(vars, p, S_dn)

    # display(heatmap(H_pn.data, c = :ice, title="Ice thickness at t=$t"))
    # display(heatmap(∇S_norm.data, c = :speed, title="∇S_norm at t=$t"))
    # display(heatmap(D_pn.data, title="D at t=$t"))

    #####################################################
    ####   2 - Update ice thicknesses on Dual Nodes  ####
    #####################################################
    #println("Calculating flux divergence on Dual Nodes \n")

    # 2.1: 2-point average of diffusivity from Primal Nodes to Dual Edges
    D_de = Edges(Dual, ∇_S_de)
    grid_interpolate!(D_de.u, avg_x(D_pn))
    grid_interpolate!(D_de.v, avg_y(D_pn))

    # 2.2: 2-point central estimate of surface gradient from Dual Nodes to Dual Edges
    ∇_S_de

    # 2.3: 2-point central estimate of flux divergence from Dual Edges to Dual Nodes
    # Computation of flux (F) on Dual Edges
    F_de = Edges(Dual, D_de)
    F_de.u .= D_de.u.*∇_S_de.u # F = D*∂S/∂x 
    F_de.v .= D_de.v.*∇_S_de.v # F = D*∂S/∂y
    
    F_dn = Nodes(Dual, S_dn)
    divergence!(F_dn, F_de/Δx) # ∂²F/∂x² + ∂²F/∂y²

    du = F_dn.data

end

###############################################################
###########################  MAIN #############################
###############################################################

# Load the HDF5 file with Harry's simulated data
root_dir = pwd()
argentiere_f = h5open(joinpath(root_dir, "data/Argentiere_2003-2100_aflow2e-16_50mres.h5"), "r")
# Fill the Glacier structure with the retrieved data
argentiere = Glacier(HDF5.read(argentiere_f["bed"])[begin:end-2,:],
                    HDF5.read(argentiere_f["thick_hist"])[begin:end-2,:,:],
                    HDF5.read(argentiere_f["vel_hist"])[begin:end-2,:,:])

# Argentière bedrock
hm01 = heatmap(argentiere.bed, c = :turku, title="Bedrock")
# Argentière ice thickness for an individual year
hm02 = heatmap(argentiere.thick[:,:,1], c = :ice, title="Ice thickness")
# Surface velocities
hm03 = heatmap(argentiere.vel[:,:,15], c = :speed, title="Ice velocities")
hm0 = plot(hm01,hm02,hm03, layout=(3,1), aspect_ratio=:equal, size=(500,1000), xlims=(0,180))
display(hm0)

# Initial parameters
p = (2e-16, 900, 9.81, 3) # A, ρ, g, n
# Spatial and temporal differentials
Δx = 50 #m (Δx = Δy)
Δt = 100 #years
H₀ = copy(argentiere.thick[:,:,1])
S₀ = copy(argentiere.bed) .+ copy(argentiere.thick[:,:,1])

u₀ = (H₀, S₀)
tspan = (0.0, 10.0)

# Create ODE problem based on the SIA
SIA_prob = ODEProblem(SIA!, u₀, tspan, p)
# Solve the SIA 
SIA_sol = solve(SIA_prob,alg_hints=[:stiff],reltol=1e-8,abstol=1e-8)


##################################################
###### Plot glacier evolution for each year  #####
##################################################
#if(t%1 == 0)
# if(true)
#     println(t)
#     hm11 = heatmap(S_dn.data, c = :turku, title="DEM")
#     hm12 = heatmap(H_dn.data, c = :ice, title="Ice thickness")
#     hm13 = heatmap(F_dn.data, title="Flux divergence")
#     hm14 = heatmap(vel_dn, c = :speed, title="Ice velocities")
#     hm1 = plot(hm11,hm12,hm13,hm14, layout=4, aspect_ratio=:equal, size=(800,600), xlims=(0,180), plot_title=t)
#     display(hm1)
# end

