#= Glacier ice dynamics toy model

Toy model based on the Shallow Ice Approximation (SIA), mixing
partial differential equations (PDEs), neural networks and model
interpretation using SINDy.
=#

## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()
using Infiltrator
using Debugger

# using Flux, DiffEqFlux, DataDrivenDiffEq
# using Flux: @epochs
# using Zygote
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

# 4 point average of Dual Nodes to a single Primal Node
function four_point_avg!(Nodes_4p_avg::Nodes{Primal}, dualNodes::Nodes{Dual}, i,j)
    Nodes_4p_avg[i,j] = (dualNodes[i,j] + dualNodes[i+1,j] + dualNodes[i,j+1] + dualNodes[i+1,j+1])/4
end

# 4 point average of Dual Edges to a single Primal Node
function four_point_avg!(Nodes_4p_avg::Nodes{Primal}, dualEdges::Edges{Dual}, i,j)
    Nodes_4p_avg[i,j] = (dualEdges.u[i,j] + dualEdges.u[i,j+1] + dualEdges.v[i,j] + dualEdges.v[i+1,j])/4
end

# Performs a 4 point average of Node values of the Dual grid to the Primal grid
function dual2primal_4p_avg!(primalNodes::Nodes{Primal}, dualNodes::Nodes{Dual})
    for i in 1:size(primalNodes)[1]
        for j in 1:size(primalNodes)[2]
            four_point_avg!(primalNodes, dualNodes, i,j)
        end
    end
end

# Performs a 4 point average of Edge values of the Dual grid to the Primal grid
function dual2primal_4p_avg!(primalNodes::Nodes{Primal}, dualEdges::Edges{Dual})
    for i in 1:size(primalNodes)[1]
        for j in 1:size(primalNodes)[2]
            four_point_avg!(primalNodes, dualEdges, i,j)
        end
    end
end

function SIA_hybrid(params, Ua, Un, Uc)


    return u
end

function SIA!(du, u, p, t)
    A, ρ, g, n = p
    H, ∂S∂x, ∂S∂y = u
    # Flux in x axis
    du[1] = ((2*A)/(n+2))*((ρ*g*H)^n)*(H^2)*∂S∂x
    # Flux in y axis
    du[2] = ((2*A)/(n+2))*((ρ*g*H)^n)*(H^2)*∂S∂y
end

# Compute diffusivities on u and v
function diffusivities(vars, p)
    A, ρ, g, n = p
    H, Su, Sv = vars
    D_u = (2/(n+2)).*((ρ.*g.*H).^n).*A.*(H.^2).*(Su.^2)
    D_v = (2/(n+2)).*((ρ.*g.*H).^n).*A.*(H.^2).*(Sv.^2)
    # Diffusivities Nodes
    D_pn_u = Nodes(Primal,dem_dn)
    D_pn_v = Nodes(Primal,dem_dn)
    D_pn_u .= D_u
    D_pn_v .= D_v

    return D_pn_u, D_pn_v
end

function u_s_test!(du, u, p, idx, t)
    A, ρ, g, n = p
    H, ∂S∂x, ∂S∂y = u
    du[idx[1], idx[2]] = abs(((2*A)/(n+2))*((ρ*g*H)^n)*H*∂S∂x*∂S∂x^(n-1)) + abs(((2*A)/(n+2))*((ρ*g*H)^n)*H*∂S∂y*∂S∂y^(n-1))
end

###############################################################
###########################  MAIN #############################
###############################################################

grid_size = 50 #m

# Load the HDF5 file with Harry's simulated data
argentiere_f = h5open("/Users/Bolib001/Desktop/Jordi/Data/Toy ice dynamics/Argentiere_2003-2100_aflow2e-16_50mres.h5", "r")
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
hm0 = plot(hm01,hm02,hm03, layout=(3,1), aspect_ratio=:equal, size=(500,1000))
display(hm0)

arg_dem = copy(argentiere.bed) .+ copy(argentiere.thick[:,:,1]) 
arg_thick = copy(argentiere.thick[:,:,1]) 
p = (2e-16, 900, 9.81, 3) # A, ρ, g, n

####################################
##### Create staggered grid  #######
####################################
# Bedrock
bed_dn = Nodes(Dual,size(argentiere.bed))
bed_dn .= argentiere.bed
bed_de = Edges(Dual, bed_dn)
bed_pn = Nodes(Primal, bed_dn)

# Ice thickness
thick_dn = Nodes(Dual,size(argentiere.thick[:,:,1]))
thick_dn .= argentiere.thick[:,:,1]
thick_de = Edges(Dual, thick_dn)
thick_pn = Nodes(Primal, thick_dn)

# DEM
dem_dn = Nodes(Dual,size(arg_dem))
dem_dn .= arg_dem
dem_pn = Nodes(Primal,dem_dn)
#dem_pn .= arg_dem[1:end-1,1:end-1,1]

# Surface gradients
grad_dn = Nodes(Dual,dem_dn)
grad_pn_u = Nodes(Primal,grad_dn)
grad_pn_v = Nodes(Primal,grad_dn)

#####################################################
#### 1 - Calculate diffusivities in Primal grid  ####
#####################################################
println("Calculating diffusivities in Primal Nodes \n")

# 1.1: 4-point average of ice thickness from Dual to Primal  
dual2primal_4p_avg!(thick_pn, thick_dn)

# 1.2: 2-point central estimate of surface gradient from Dual Nodes to Dual Edges
grad_de = grad(dem_dn)

# 1.3: 2-point average of gradients from Dual Edges to Primal Nodes
#dual2primal_4p_avg!(grad_pn, grad_de)
grid_interpolate!(grad_pn_u, grad_de.u)
grid_interpolate!(grad_pn_v, grad_de.v)

# 1.4: Calculate diffusivities on Primal Nodes
vars = thick_pn, grad_pn_u, grad_pn_v
D_pn_u, D_pn_v = diffusivities(vars, p)

#####################################################
####   2 - Update ice thicknesses on Dual Nodes  ####
#####################################################
println("Calculating flux divergence on Dual Nodes \n")

# 2.1: 2-point average of diffusivity from Primal Nodes to Dual Edges
D_de = Edges(Dual, dem_dn)
grid_interpolate!(D_de.u, D_pn_u)
grid_interpolate!(D_de.v, D_pn_v)

# 2.2: 2-point central estimate of surface gradient from Dual Nodes to Dual Edges
grad_de

# 2.3: 2-point central estimate of flux divergence from Dual Edges to Dual Nodes
# Computation of flux (F) on Dual Edges
F_de = Edges(Dual, D_de)
F_e_u = D_de.u.*grad_de.u # F = D*∂h/∂x 
F_e_v = D_de.v.*grad_de.v # F = D*∂h/∂y
F_de.u .= F_e_u
F_de.v .= F_e_v
# Computation of flux divergence on Dual Nodes
divergence!(F_dn, F_de)
#vel_dn = Nodes(Dual, F_dn)
vel_dn = F_dn.data./thick_dn.data

hm11 = heatmap(F_dn.data, title="Flux divergence")
hm12 = heatmap(thick_dn.data, c = :ice, title="Ice thickness")
hm13 = heatmap(vel_dn, c = :speed, title="Ice velocities")
hm1 = plot(hm11,hm12,hm13, layout=(3,1), aspect_ratio=:equal, size=(500,1000))
display(hm1)

### Apply flux on ice thicknesses  ####

thick_dn += F_dn
dem_dn += F_dn

# ∇_diffusion = laplacian(dem_dn)
# S_dem = grad(dem_dn)
# println("S_dem: ", S_dem)
# D = u_s(thick_de, p).*(S_dem.^2)
# #display(heatmap(D.*∇_diffusion))

# p_∇ = plot(heatmap(∇_diffusion))
# p_grad = plot(heatmap(S_dem))
# println("Size D: ", size(D))
# println("Size ∇_diffusion: ", size(∇_diffusion))
# #plot_∇ = plot(p_∇, p_grad, layout=l1)
# #display(p_grad)

# dem_pn += D.*∇_diffusion
# thick_pn += D.*∇_diffusion
# #println("Year: ", year)
# hm = heatmap(arg_thick, clim=(0,350))
# #hm = heatmap(arg_dem, clim=(1000,4000))
# display(hm)



